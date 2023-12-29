import argparse, random, os
from typing import List, Tuple

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import common.mpi_utils as mpi
from common.policy import CategoricalPolicy, DiagonalGaussianPolicy
from common.vf import StateValueFunction
from common.utils import set_seed, to_tensor, dim
from common.buffer import RolloutBuffer
from common.logger import Logger


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
    :param seed: (int) Seed for RNG
    :param hidden_sizes: (List[int]) Sizes of policy & Q networks' hidden layers
    :param pi_lr: (float )Learning rate for policy opitimizer
    :param v_lr: (float) Learning rate for value function optimizer
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
    :param save: (bool) Whether to save the final model.
    :param save_every: (int) Model saving frequency.
    :param render: (bool) Whether to render the training result.
    :param plot: (bool) Whether to plot the statistics.
    """

    def __init__(self, args) -> None:
        self.env = gym.make(args.env)
        set_seed(args.seed + 10 * mpi.proc_rank())
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.device = 'cuda:0' if torch.cuda.is_available() and args.cpu == 1 else 'cpu'
        self.ac = ActorCritic(observation_space, action_space, args.hidden_sizes, device=self.device)
        mpi.sync_params(self.ac)
        self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.pi_lr)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
        self.epochs = args.epochs
        self.proc_steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
        self.steps_per_epoch = args.steps_per_epoch
        self.train_pi_iters = args.train_pi_iters
        self.train_v_iters = args.train_v_iters
        self.max_ep_len = args.max_ep_len
        self.buffer = RolloutBuffer(self.proc_steps_per_epoch, args.gamma, args.lamb)
        self.kl_target = args.kl_target
        if args.clip:
            self.clip_ratio = args.clip_ratio
        else:
            pass
        self.save = args.save
        self.save_every = args.save_every
        self.render = args.render
        self.plot = args.plot
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        config_dict = vars(args)
        config_dict['algo'] = 'ppo'
        self.logger = Logger(log_dir=log_dir)
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.ac)


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
            v_values = self.ac.critic(observations)
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


    def select_action(self, observation: np.ndarray, action_only: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation = to_tensor(observation, device=self.device)
        with torch.no_grad():
            pi = self.ac.actor._distribution(observation)
            action = pi.sample()
            if action_only:
                return action.cpu().numpy()
            log_prob = self.ac.actor._log_prob(pi, action)
            value = self.ac.critic(observation)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


    def load(self, model_path: str) -> None:
        """Model loading"""
        self.ac.load_state_dict(torch.load(model_path))


    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            step = 0
            while step < self.proc_steps_per_epoch:
                observation, _ = self.env.reset()
                rewards = []

                while True:
                    action, log_prob, value = self.select_action(observation, action_only=False)
                    next_observation, reward, terminated, truncated, _ = self.env.step(action)
                    self.buffer.add(observation, action, reward, value.item(), log_prob.item())
                    observation = next_observation
                    rewards.append(reward)
                    step += 1

                    if terminated or truncated or (len(rewards) == self.max_ep_len) \
                            or (step == self.proc_steps_per_epoch):
                        if terminated or truncated:
                            value = 0
                            self.logger.add({
                                'episode-return': sum(rewards),
                                'episode-length': len(rewards)
                            })
                        else:
                            _, _, value = self.select_action(observation, action_only=False)
                        self.buffer.finish_rollout(value)
                        break
            self.update_params()

            self.logger.log_epoch('epoch', epoch)
            self.logger.log_epoch('pi-loss', average_only=True)
            self.logger.log_epoch('v-loss', average_only=True)
            self.logger.log_epoch('v-values', need_optima=True)
            self.logger.log_epoch('kl', average_only=True)
            self.logger.log_epoch('episode-return', need_optima=True)
            self.logger.log_epoch('episode-length', average_only=True)
            self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
            self.logger.dump_epoch()

            if self.save and epoch % self.save_every == 0:
                self.logger.save_state()
        self.env.close()
        if self.render:
            self.logger.render(self.select_action)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='ppo',
                        help='Experiment name')
    parser.add_argument('--cpu', type=int, default=4,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 32],
                        help="Sizes of policy & Q networks' hidden layers")
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--v-lr', type=float, default=1e-3,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
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
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the final model')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the training result')
    parser.add_argument('--plot', action='store_true',
                        help=' Whether to plot the statistics')
    args = parser.parse_args()
    if args.clip and not args.clip_ratio:
        parser.error('Argument --clip-ratio is required when --clip is enabled.')
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = PPO(args)
    agent.train()
