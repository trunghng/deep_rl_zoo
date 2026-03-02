import argparse, random, os
from collections import deque
from typing import List, Tuple

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
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
from common.utils import set_seed, to_tensor, dim
from common.buffer import VectorRolloutBuffer
from common.logger import Logger
from common.lr_schedulers import get_linear_scheduler
import envs


class ActorCritic(nn.Module):
    
    def __init__(self,
                obs_space: Space,
                action_space: Space,
                hidden_sizes: List[int]=[64, 64],
                activation: nn.Module=nn.Tanh,
                device: str='cpu') -> None:
        super().__init__()
        obs_dim = dim(obs_space)
        action_dim = dim(action_space)

        # continous action space
        if isinstance(action_space, Box):
            self.actor = DiagonalGaussianPolicy(obs_dim, action_dim, hidden_sizes, activation).to(device)
        # discrete action space
        elif isinstance(action_space, Discrete):
            self.actor = CategoricalPolicy(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic = StateValueFunction(obs_dim, hidden_sizes, activation).to(device)


class PPO:
    """
    PPO w/ Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function

    :param env: (str) Environment name
    :param exp_name: (str) Experiment name
    :param cpu: (int) Number of CPUs for parallel computing
    :param n_envs: (int) The number of sub-environment in the vector environment
    :param seed: (int) Seed for RNG
    :param hidden_sizes: (List[int]) Sizes of policy & Q networks' hidden layers
    :param pi_lr: (float )Learning rate for policy opitimizer
    :param v_lr: (float) Learning rate for value function optimizer
    :param lr_decay: (bool) Use linear LR decay
    :param epochs: (int) Number of epochs
    :param steps_per_epoch: (int) Maximum number of steps per epoch
    :param train_pi_iters: (int) Number of GD-steps to take on policy loss per epoch
    :param train_v_iters: (int) Number of GD-steps to take on value function per epoch
    :param max_ep_len: (int) Maximum episode/trajectory length
    :param gamma: (float) Discount factor
    :param lamb: (float) Lambda for GAE
    :param kl_target: (float) KL divergence threshold
    :param clip: (bool) Whether to use clipping, enable penalty otherwise
    :param clip_ratio: (float) Hyperparamter for clipping the policy objective
    :param norm_obs: (bool) Normalize observations
    :param norm_rew: (bool) Normalize rewards
    :param resume: (str) Path to checkpoint to continue training from
    :param save: (bool) Save the final model.
    :param save_every: (int) Model saving frequency.
    :param save_latest_only: (bool) Save the latest model only
    :param render: (bool) Render the training result.
    :param plot: (bool) Plot the statistics.
    """

    def __init__(self, args) -> None:
        self.envs = gym.make_vec(
            args.env,
            num_envs=args.n_envs,
            vectorization_mode='async',
            wrappers=[lambda env: ClipAction(env)]
        )
        if args.norm_obs:
            self.envs = NormalizeObservation(self.envs)
        if args.norm_rew:
            self.envs = NormalizeReward(self.envs, gamma=args.gamma)
        self.envs = RecordEpisodeStatistics(self.envs)
        set_seed(args.seed + 10 * mpi.proc_rank())

        if hasattr(self.envs, 'single_observation_space'):
            observation_space = self.envs.single_observation_space
            action_space = self.envs.single_action_space
        else:
            observation_space = self.envs.observation_space
            action_space = self.envs.action_space

        self.n_envs = args.n_envs
        self.device = 'cuda:0' if torch.cuda.is_available() and args.cpu == 1 else 'cpu'
        self.ac = ActorCritic(observation_space, action_space, args.hidden_sizes, device=self.device)
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
        self.epochs = args.epochs
        self.proc_steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
        self.steps_per_epoch = args.steps_per_epoch
        self.train_pi_iters = args.train_pi_iters
        self.train_v_iters = args.train_v_iters
        self.max_ep_len = args.max_ep_len
        self.buffer = VectorRolloutBuffer(
            obs_dim=dim(observation_space),
            action_dim=dim(action_space),
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
        self.enable_save = args.save
        self.save_every = args.save_every
        self.save_latest_only = args.save_latest_only
        self.render = args.render
        self.plot = args.plot

        wandb_id = None
        self.start_epoch = 1
        if args.resume:
            checkpoint = self.load_checkpoint(args.resume)
            wandb_id = checkpoint.get('wandb_id')
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            mpi.mpi_print(f'Successfully loaded checkpoint from epoch {self.start_epoch - 1}')

            if self.use_lr_decay:
                for _ in range(self.start_epoch - 1):
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()

        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        config_dict = vars(args)
        config_dict['algo'] = 'ppo'
        config_dict['wandb_id'] = wandb_id
        config_dict['device'] = self.device
        self.logger = Logger(log_dir=log_dir, config=config_dict)
        self.logger.save_config(config_dict)
        if args.resume:
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

            # approximated avg KL
            # approx_kl = (-log_ratio).mean().item()
            # http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            return pi_loss, approx_kl

        def compute_v_loss(observations, rewards_to_go):
            v_values = self.ac.critic(observations).squeeze(-1)
            v_loss = ((v_values - rewards_to_go) ** 2).mean()
            return v_loss, v_values

        observations, actions, logps_prob, advs, rewards_to_go \
            = map(lambda x: x.to(self.device), self.buffer.get())

        for step in range(1, self.train_pi_iters + 1):
            self.actor_opt.zero_grad()
            pi_loss, approx_kl = compute_pi_loss(observations, actions, logps_prob, advs)
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
            'kl': kl
        })

    def select_action(self, observations: np.ndarray, action_only: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = to_tensor(observations, device=self.device)
        with torch.no_grad():
            pi = self.ac.actor._distribution(observations)
            actions = pi.sample()
            if action_only:
                return actions.cpu().numpy()
            log_probs = self.ac.actor._log_prob(pi, actions)
            values = self.ac.critic(observations)
        return actions.cpu().numpy(), log_probs.cpu().numpy().flatten(), values.cpu().numpy().flatten()

    def save(self, epoch: int):
        obs_rms = None
        current_env = self.envs
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'obs_rms'):
                obs_rms = current_env.obs_rms
                break
            current_env = current_env.env
        state = {
            'epoch': epoch,
            'actor': self.ac.actor.state_dict(),
            'critic': self.ac.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'obs_rms': obs_rms,
            'wandb_id': wandb.run.id if (self.logger.use_wandb and mpi.proc_rank() == 0) else None
        }
        self.logger.save_latest(state)
        if not self.save_latest_only and epoch % self.save_every == 0:
            self.logger.save_state(state, epoch)

    def load(self, model_path: str) -> None:
        """Lightweight load for inference only"""
        checkpoint = torch.load(model_path)
        if 'actor' in checkpoint:
            self.ac.actor.load_state_dict(checkpoint['actor'])
        else:
            # Backward compatibility for old model files
            self.ac.load_state_dict(checkpoint)

    def load_checkpoint(self, checkpoint_path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ac.actor.load_state_dict(checkpoint['actor'])
        self.ac.critic.load_state_dict(checkpoint['critic'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])

        if 'obs_rms' in checkpoint and checkpoint['obs_rms']:
            mpi.mpi_print('Restoring observation normalization statistics...')
            current_env = self.envs
            while hasattr(current_env, 'env'):
                if hasattr(current_env, 'obs_rms'):
                    current_env.obs_rms = checkpoint['obs_rms']
                    break
                current_env = current_env.env
        return checkpoint

    def train(self) -> None:
        observations, _ = self.envs.reset() # (B, O)
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                epoch_returns = []
                epoch_lengths = []
                rewards_upright, rewards_energy, forward_velocities, torso_heights = [], [], [], []

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

                    if 'reward_upright' in info:
                        rewards_upright.append(info['reward_upright'].mean())
                        rewards_energy.append(info['reward_energy'].mean())
                        forward_velocities.append(info['forward_velocity'].mean())
                        torso_heights.append(info['torso_height'].mean())

                self.logger.add({
                    'episode-return': epoch_returns,
                    'episode-length': epoch_lengths
                })
                if rewards_upright:
                    self.logger.add({
                        'env-reward-upright': rewards_upright,
                        'env-reward-energy': rewards_energy,
                        'env-forward-velocity': forward_velocities,
                        'env-torso-height': torso_heights
                    })

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
                if rewards_upright:
                    self.logger.log_epoch('env-reward-upright', average_only=True)
                    self.logger.log_epoch('env-reward-energy', average_only=True)
                    self.logger.log_epoch('env-forward-velocity', average_only=True)
                    self.logger.log_epoch('env-torso-height', average_only=True)
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
    parser.add_argument('--norm-obs', action='store_true',
                        help='Normalize observations')
    parser.add_argument('--norm-rew', action='store_true',
                        help='Normalize rewards')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
    args = parser.parse_args()
    if args.clip and not args.clip_ratio:
        parser.error('Argument --clip-ratio is required when --clip is enabled.')
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = PPO(args)
    agent.train()
