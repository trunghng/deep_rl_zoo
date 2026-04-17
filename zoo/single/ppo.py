import argparse, random, os
from collections import defaultdict
from typing import List, Tuple

import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.wrappers import ClipAction
from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward, RecordEpisodeStatistics
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import wandb

import common.mpi_utils as mpi
from common.policy import CategoricalPolicy, DiagonalGaussianPolicy
from common.vf import StateValueFunction
from common.utils import set_seed, to_tensor, dim, get_spaces, get_base_env
from common.buffer import VectorRolloutBuffer
from common.logger import Logger
from common.lr_schedulers import get_linear_scheduler
import envs
import envs.base.base_config as base_config


class ActorCritic(nn.Module):
    
    def __init__(self,
                obs_dim: int,
                act_dim: int,
                is_discrete_action: bool,
                hidden_sizes: List[int]=[64, 64],
                activation: nn.Module=nn.Tanh,
                device: str='cpu') -> None:
        super().__init__()

        # continous action space
        if not is_discrete_action:
            self.actor = DiagonalGaussianPolicy(obs_dim, act_dim, hidden_sizes, activation).to(device)
        # discrete action space
        else:
            self.actor = CategoricalPolicy(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.critic = StateValueFunction(obs_dim, hidden_sizes, activation).to(device)


class PPO:
    """
    PPO w/ Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function
    """

    def __init__(self, args) -> None:
        self.test_mode = getattr(args, 'test_mode', False)
        self.envs = None
        wandb_id = None

        if self.test_mode and hasattr(args, 'obs_dim') and hasattr(args, 'act_dim'):
            self.device = 'cuda:0' if torch.cuda.is_available() and getattr(args, 'cpu', 1) == 1 else 'cpu'
            self.ac = ActorCritic(
                obs_dim=args.obs_dim,
                act_dim=args.act_dim,
                is_discrete_action=getattr(args, 'is_discrete_action', False),
                hidden_sizes=args.hidden_sizes,
                device=self.device
            )
        else:
            spec = gym.spec(args.env)
            self.is_custom_env = spec.entry_point.startswith('envs.')
            env_kwargs = {}
            vec_mode = 'async'
            if self.is_custom_env:
                for key in ['use_camera', 'use_privileged', 'use_grid']:
                    val = getattr(args, key, None)
                    if val is not None:
                        env_kwargs[key] = val
                    if hasattr(args, key):
                        delattr(args, key)

                config = base_config.BaseLeggedConfig()
                if env_kwargs.get('use_camera', config.sensor.depth_camera.enabled):
                    vec_mode = 'sync'

            self.envs = gym.make_vec(
                args.env,
                num_envs=args.n_envs,
                vectorization_mode=vec_mode,
                wrappers=[lambda env: ClipAction(env)],
                **env_kwargs
            )

            if args.norm_obs:
                self.envs = NormalizeObservation(self.envs)
            if args.norm_rew:
                self.envs = NormalizeReward(self.envs, gamma=args.gamma)
            self.envs = RecordEpisodeStatistics(self.envs)
            set_seed(args.seed + 10 * mpi.proc_rank())

            observation_space, action_space = get_spaces(self.envs)
            obs_dim, act_dim = dim(observation_space), dim(action_space)
            self.device = 'cuda:0' if torch.cuda.is_available() and args.cpu == 1 else 'cpu'

            self.ac = ActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                is_discrete_action=isinstance(action_space, Discrete),
                hidden_sizes=args.hidden_sizes,
                device=self.device
            )
            mpi.sync_params(self.ac)
            self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.pi_lr)
            self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
            self.use_lr_decay = args.lr_decay
            if self.use_lr_decay:
                self.actor_scheduler = get_linear_scheduler(
                    self.actor_opt, warmup_steps=args.epochs // 20, training_steps=args.epochs
                )
                self.critic_scheduler = get_linear_scheduler(
                    self.critic_opt, warmup_steps=args.epochs // 20, training_steps=args.epochs
                )
            else:
                self.actor_scheduler = None
                self.critic_scheduler = None

            self.start_epoch = 1
            checkpoint_path = args.resume or args.fork or args.rewind
            parent_wandb_id = None

            if checkpoint_path:
                checkpoint = self.load(checkpoint_path)
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                loaded_wandb_id = checkpoint.get('wandb_id')

                if args.resume:
                    wandb_id = loaded_wandb_id
                elif args.fork or args.rewind:
                    wandb_id = None
                    parent_wandb_id = loaded_wandb_id
                mpi.mpi_print(f'Successfully loaded checkpoint from epoch {self.start_epoch - 1}')

                if self.use_lr_decay:
                    for _ in range(self.start_epoch - 1):
                        self.actor_scheduler.step()
                        self.critic_scheduler.step()
            self.epochs = args.epochs
            self.proc_steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
            self.steps_per_epoch = args.steps_per_epoch
            self.train_pi_iters = args.train_pi_iters
            self.train_v_iters = args.train_v_iters
            self.max_ep_len = args.max_ep_len
            self.buffer = VectorRolloutBuffer(
                obs_dim=obs_dim,
                action_dim=act_dim,
                size=self.proc_steps_per_epoch,
                n_envs=args.n_envs,
                gamma=args.gamma,
                lamb=args.lamb
            )
            self.kl_target = args.kl_target
            if args.clip:
                self.clip_ratio = args.clip_ratio
            else:
                pass
            self.ent_coef = args.ent_coef
            self.enable_save = args.save
            self.save_every = args.save_every
            self.save_latest_only = args.save_latest_only
            self.render = args.render
            self.plot = args.plot

        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        config_dict = vars(args)
        config_dict['algo'] = 'ppo'
        config_dict['wandb_id'] = wandb_id
        if not self.test_mode:
            config_dict['start_epoch'] = self.start_epoch
            config_dict['device'] = self.device
            if parent_wandb_id:
                config_dict['parent_wandb_id'] = parent_wandb_id 
                if args.rewind:
                    config_dict['run_type'] = 'rewind'
                elif args.fork:
                    config_dict['run_type'] = 'fork'
        self.logger = Logger(log_dir=log_dir, config=config_dict)
        self.logger.save_config(config_dict, env=self.envs)
        if args.resume and not self.test_mode:
            self.logger.truncate_log(self.start_epoch - 1)

    def update_params(self) -> None:
        """Update policy & value networks' parameters"""

        def compute_pi_loss(observations, actions, logps_old, advs):
            pi, logps = self.ac.actor(observations, actions)
            log_ratio = logps - logps_old
            ratio = log_ratio.exp()
            loss_cpi = ratio * advs
            clip_advs = ((1 + self.clip_ratio) * (advs > 0) + (1 - self.clip_ratio) * (advs < 0)) * advs

            pi_loss = -torch.min(loss_cpi, clip_advs).mean()
            entropy = pi.entropy().mean()
            total_pi_loss = pi_loss - self.ent_coef * entropy

            # approximated avg KL
            # approx_kl = (-log_ratio).mean().item()
            # http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            return total_pi_loss, approx_kl, entropy.item()

        def compute_v_loss(observations, rewards_to_go):
            v_values = self.ac.critic(observations).squeeze(-1)
            v_loss = ((v_values - rewards_to_go) ** 2).mean()
            return v_loss, v_values

        observations, actions, logps_prob, advs, rewards_to_go \
            = map(lambda x: x.to(self.device), self.buffer.get())

        for step in range(1, self.train_pi_iters + 1):
            self.actor_opt.zero_grad()
            pi_loss, approx_kl, entropy = compute_pi_loss(observations, actions, logps_prob, advs)
            kl = mpi.mpi_avg(approx_kl)
            if kl > 1.5 * self.kl_target:
                self.logger.log(f'Early stopping at step {step} due to exceeding KL target')
                break
            pi_loss.backward()
            mpi.mpi_avg_grads(self.ac.actor)
            self.actor_opt.step()

        for _ in range(self.train_v_iters):
            self.critic_opt.zero_grad()
            v_loss, v_values = compute_v_loss(observations, rewards_to_go)
            v_loss.backward()
            mpi.mpi_avg_grads(self.ac.critic)
            self.critic_opt.step()

        self.logger.add({
            'pi-loss': pi_loss.item(),
            'v-loss': v_loss.item(),
            'v-values': v_values.detach().cpu().numpy(),
            'kl': kl,
            'entropy': entropy
        })

    def select_action(
        self,
        observations: np.ndarray,
        action_only: bool = True,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = to_tensor(observations, device=self.device)
        with torch.no_grad():
            pi = self.ac.actor._distribution(observations)
            actions = pi.mean if deterministic else pi.sample()
            if action_only:
                return actions.cpu().numpy()
            log_probs = self.ac.actor._log_prob(pi, actions)
            values = self.ac.critic(observations)
        return actions.cpu().numpy(), log_probs.cpu().numpy().flatten(), values.cpu().numpy().flatten()

    def save(self, epoch: int):
        obs_rms, return_rms = None, None
        current_env = self.envs
        while True:
            if hasattr(current_env, 'obs_rms'):
                obs_rms = current_env.obs_rms
            if hasattr(current_env, 'return_rms'):
                return_rms = current_env.return_rms
            if not hasattr(current_env, 'env'):
                break
            current_env = current_env.env

        if self.is_custom_env:
            terrain_types = current_env.get_attr('config')[0].terrain.terrain_types
            all_env_levels = current_env.get_attr('terrain_levels')
            
            local_avg = {t: 0.0 for t in terrain_types}
            for env_dict in all_env_levels:
                for t, level in env_dict.items():
                    local_avg[t] += level / len(all_env_levels)

            terrain_levels = {t: 0.0 for t in local_avg.keys()}
            for t in local_avg.keys():
                terrain_levels[t] = mpi.mpi_avg(local_avg[t])

        state = {
            'epoch': epoch,
            'actor': self.ac.actor.state_dict(),
            'critic': self.ac.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'obs_rms': obs_rms,
            'return_rms': return_rms,
            'terrain_levels': terrain_levels,
            'wandb_id': wandb.run.id if (self.logger.use_wandb and mpi.proc_rank() == 0) else None
        }
        self.logger.save_latest(state)
        if not self.save_latest_only and epoch % self.save_every == 0:
            self.logger.save_state(state, epoch)

    def load(self, checkpoint_path: str, env=None) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.ac.actor.load_state_dict(checkpoint['actor'])
        self.ac.critic.load_state_dict(checkpoint['critic'])
        if not self.test_mode:
            self.actor_opt.load_state_dict(checkpoint['actor_opt'])
            self.critic_opt.load_state_dict(checkpoint['critic_opt'])

        target_env = env if env is not None else self.envs

        if target_env:
            terrain_levels = checkpoint.get('terrain_levels')

            if terrain_levels is not None and not self.test_mode:
                mpi.mpi_print(f'Restoring terrain levels: {terrain_levels}')

            current_env = target_env
            while True:
                if terrain_levels is not None and not self.test_mode:
                    if hasattr(current_env, 'set_attr'):
                        # Distribute the population across levels based on the saved average
                        # e.g., if average is 1.25, 75% of envs start at Level 1, 25% at Level 2
                        num_envs = self.envs.num_envs
                        env_levels_list = []
                        for i in range(num_envs):
                            env_levels = {}
                            for t, avg_val in terrain_levels.items():
                                # We distribute them such that the mean remains correct
                                # Env index i: if i / num_envs < (avg % 1), we round up
                                threshold = avg_val % 1
                                level = int(avg_val) + (1 if (i / num_envs) < threshold else 0)
                                env_levels[t] = level
                            env_levels_list.append(env_levels)
                        
                        current_env.set_attr('terrain_levels', env_levels_list)
                    elif hasattr(current_env, 'terrain_levels'):
                        current_env.terrain_levels = {t: int(v) for t, v in terrain_levels.items()}

                # Restore normalization statistics
                if 'obs_rms' in checkpoint and hasattr(current_env, 'obs_rms'):
                    mpi.mpi_print('Restoring observation normalization statistics...')
                    current_env.obs_rms = checkpoint['obs_rms']
                if 'return_rms' in checkpoint and hasattr(current_env, 'return_rms'):
                    mpi.mpi_print('Restoring reward normalization statistics...')
                    current_env.return_rms = checkpoint['return_rms']
                
                if not hasattr(current_env, 'env'):
                    break
                current_env = current_env.env
        return checkpoint

    def train(self) -> None:
        observations, _ = self.envs.reset() # (B, O)
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                total_steps = (epoch - 1) * self.steps_per_epoch
                epoch_returns, epoch_lengths = [], []
                env_info_dict = defaultdict(list)

                for t in range(self.proc_steps_per_epoch):
                    actions, log_probs, values = self.select_action(observations, action_only=False) # (B, A), (B,), (B,)
                    next_observations, rewards, terminations, truncations, info = self.envs.step(actions) # (B, O), (B,), (B,), (B,)
                    self.buffer.add(observations, actions, rewards, values, log_probs, terminations, truncations)
                    observations = next_observations # (B, O)

                    if 'episode' in info:
                        mask = info['_episode']
                        episode_returns = info['episode']['r']
                        episode_lengths = info['episode']['l']

                        for i in range(len(mask)):
                            if mask[i]:
                                epoch_returns.append(episode_returns[i])
                                epoch_lengths.append(episode_lengths[i])

                    for key, val in info.items():
                        if key.startswith('reward_') or key in ['forward_velocity', 'torso_height']:
                            log_key = f"env-{key.replace('_', '-')}"
                            env_info_dict[log_key].append(val.mean())

                self.logger.add({
                    'episode-return': epoch_returns,
                    'episode-length': epoch_lengths
                })
                if env_info_dict:
                    self.logger.add(env_info_dict)

                _, _, last_values = self.select_action(observations, action_only=False)
                self.buffer.finish_rollout(last_values)
                self.update_params()

                self.logger.log_epoch('epoch', epoch)

                if self.use_lr_decay:
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()
                    current_pi_lr = self.actor_opt.param_groups[0]['lr']
                    current_v_lr = self.critic_opt.param_groups[0]['lr']
                    self.logger.log_epoch('pi-lr', current_pi_lr)
                    self.logger.log_epoch('v-lr', current_v_lr)

                self.logger.log_epoch('pi-loss', average_only=True)
                self.logger.log_epoch('v-loss', average_only=True)
                self.logger.log_epoch('v-values', need_optima=True)
                self.logger.log_epoch('kl', average_only=True)
                self.logger.log_epoch('entropy', average_only=True)

                if self.is_custom_env:
                    base_env = get_base_env(self.envs)
                    terrain_types = base_env.get_attr('config')[0].terrain.terrain_types
                    all_env_levels = base_env.get_attr('terrain_levels')
                    
                    local_avg = {t: 0.0 for t in terrain_types}
                    for env_dict in all_env_levels:
                        for t, level in env_dict.items():
                            local_avg[t] += level / len(all_env_levels)

                    for t in local_avg.keys():
                        global_avg = mpi.mpi_avg(local_avg[t])
                        self.logger.log_epoch(f"level-{t.replace('_', '-')}", global_avg)

                if env_info_dict:
                    for key in env_info_dict.keys():
                        self.logger.log_epoch(key, average_only=True)
                self.logger.log_epoch('episode-return', need_optima=True)
                self.logger.log_epoch('episode-length', average_only=True)
                self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
                self.logger.dump_epoch()

                if self.render and epoch % 10 == 0:
                    self.logger.log(f'🎬 Rendering live evaluation at epoch {epoch}...')
                    get_action = lambda obs: self.select_action(np.expand_dims(obs, axis=0), action_only=True)[0]
                    self.logger.render(get_action, video=False)

                if self.enable_save:
                    self.save(epoch)
            if self.render:
                self.logger.log('🎬 Saving final evaluation video...')
                get_action = lambda obs: self.select_action(np.expand_dims(obs, axis=0), action_only=True)[0]
                self.logger.render(get_action, video=True)

            if self.plot:
                self.logger.plot()
        except Exception as e:
            self.logger.log(f'Training interrupted by error: {e}')
            raise e
        finally:
            self.logger.log('🧹 Cleaning up environment and logger...')
            self.envs.close()
            self.logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='ppo',
                        help='Experiment name')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='The number of sub-environment in the vector environment')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 32],
                        help="Sizes of policy & Q networks' hidden layers")
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--v-lr', type=float, default=1e-3,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--lr-decay', action='store_true',
                        help='Use linear LR decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=2048,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--train-pi-iters', type=int, default=80,
                        help='Number of gradient descent steps to take on policy loss per epoch')
    parser.add_argument('--train-v-iters', type=int, default=80,
                        help='Number of gradient descent steps to take on value function per epoch')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float, default=0.97,
                        help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--kl-target', type=float, default=0.01,
                        help='KL divergence threshold')
    parser.add_argument('--clip', action='store_false',
                        help='Whether to use PPO-Clip, use PPO-Penalty otherwise')
    parser.add_argument('--clip-ratio', type=float, default=0.2,
                        help='Hyperparameter for clipping in the policy objective')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')
    parser.add_argument('--norm-obs', action='store_true',
                        help='Normalize observations')
    parser.add_argument('--norm-rew', action='store_true',
                        help='Normalize rewards')
    parser.add_argument('--use-camera', action='store_true',
                        help='Enable depth camera')
    parser.add_argument('--use-privileged', action='store_true',
                        help='Enable privileged info')
    parser.add_argument('--use-grid', action='store_true',
                        help='Enable grid map mode')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--save', action='store_true',
                        help='Save the final model')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Model saving frequency')
    parser.add_argument('--save-latest-only', action='store_true',
                        help='Save the latest model only')
    parser.add_argument('--render', action='store_true',
                        help='Render the training result')
    parser.add_argument('--plot', action='store_true',
                        help=' Plot the statistics')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--resume', type=str, default=None,
                                  help='Path to checkpoint to resume from (continues the same W&B run)')
    checkpoint_group.add_argument('--fork', type=str, default=None,
                                  help='Path to checkpoint to fork from (starts a new W&B run linked to the old one)')
    checkpoint_group.add_argument('--rewind', type=str, default=None,
                                  help='Path to checkpoint to rewind from (erases W&B history after this point)')
    args = parser.parse_args()
    if args.clip and not args.clip_ratio:
        parser.error('Argument --clip-ratio is required when --clip is enabled.')
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = PPO(args)
    agent.train()
