import argparse, random, os
from typing import List
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.autograd import grad
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from common.policy import DeterministicPolicy, DiscretePolicy
from common.vf import StateActionValueFunction
from common.utils import soft_update, dim, set_seed, to_tensor
from common.buffer import ReplayBuffer
from common.logger import Logger


class ActorCritic(nn.Module):


    def __init__(self,
                observation_space: Space,
                action_space: Space,
                hidden_sizes: List[int]=[256, 256],
                activation=nn.ReLU,
                device: str='cpu') -> None:
        super().__init__()
        obs_dim = dim(observation_space)
        action_dim = dim(action_space)

        # continuous action space
        if isinstance(action_space, Box):
            action_limit = action_space.high[0]
            self.actor = DeterministicPolicy(obs_dim, action_dim, hidden_sizes, 
                            activation, nn.Tanh, action_limit).to(device)
            self.critic = StateActionValueFunction(obs_dim, action_dim, hidden_sizes, 
                            activation).to(device)
        # discrete action space
        elif isinstance(action_space, Discrete):
            self.actor = DiscretePolicy(obs_dim, action_dim, hidden_sizes, 
                            activation, nn.Identity).to(device)
            self.critic = StateActionValueFunction(obs_dim, 1, hidden_sizes, activation).to(device)


class DDPG:
    """
    Deep Determinisitic Policy Gradient

    :param env: (str) Environment ID.
    :param exp_name: (str) Experiment name.
    :param seed: (int) Seed for RNG.
    :param hidden_sizes: (List[int]) Sizes of policy & Q networks' hidden layers
    :param mu_lr: (float) Learning rate for policy optimizer.
    :param q_lr: (float) Learning rate for value function optimizer.
    :param lambd: (float) Softmax temperature parameter, used in Gumbel-Softmax trick
    :param epochs: (int) Number of epochs.
    :param steps_per_epoch: (int) Maximum number of steps per epoch.
    :param max_ep_len: (int) Maximum length of an episode.
    :param buffer_size: (int) Replay buffer size.
    :param batch_size: (int) Minibatch size.
    :param start_step: (int) Start step to begin select action according to policy network.
    :param update_every: (int) Parameters update frequency.
    :param update_after: (int) Number of steps after which paramters update is allowed. 
                This guarantees there are enough number of training experience in the replay buffer.
    :param gamma: (float) Discount factor
    :param tau: (float) Soft (Polyak averaging) update coefficient
    :param sigma: (float) Standard deviation of mean-zero Gaussian noise for exploration.
                The original DDPG used Ornstein-Uhlenbeck process instead.
    :param test_episodes: (int) Number of episodes to test the deterministic policy at the end of each episode.
    :param save: (bool) Whether to save the final model.
    :param save_every: (int) Model saving frequency.
    :param render: (bool) Whether to render the training result.
    :param plot: (bool) Whether to plot the statistics.
    """

    def __init__(self, args) -> None:
        self.env = gym.make(args.env)
        set_seed(args.seed)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.action_dim = dim(action_space)
        if isinstance(action_space, Discrete):
            self.is_continuous = False
            self.lambd = args.lambd
        else:
            self.action_limit = action_space.high[0]
            self.is_continuous = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ac = ActorCritic(observation_space, action_space, args.hidden_sizes, device=self.device)
        self.ac_target = deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False
        self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.mu_lr)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.q_lr)
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.max_ep_len = args.max_ep_len
        self.buffer = ReplayBuffer(args.buffer_size)
        self.batch_size = args.batch_size
        self.start_step = args.start_step
        self.update_every = args.update_every
        self.update_after = args.update_after
        self.gamma = args.gamma
        self.tau = args.tau
        self.sigma = args.sigma
        self.test_episodes = args.test_episodes
        self.save = args.save
        self.save_every = args.save_every
        self.render = args.render
        self.plot = args.plot
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        self.logger = Logger(log_dir=log_dir)
        config_dict = vars(args)
        config_dict['algo'] = 'ddpg'
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.ac)


    def update_params(self) -> None:
        """Update policy and value networks' parameters"""

        def compute_mu_loss(observations):
            if self.is_continuous:
                actions_ = self.ac.actor(observations)
            else:
                '''
                Gumbel-Softmax trick: (arXiv:1611.00712 version)
                    logits \in (-infty, infty)
                    -> softmax(logits) \in (0, 1)
                    -> -log(softmax(logits)) \in (0, infty)
                Considering -log(softmax(logits)) as alphas
                Sampling G = -log(-log(U)) for U ~ Unif(0, 1)
                
                Gumbel-Max: a = argmax_i(log(alpha_i) + G_i)
                Gumbel-Softmax: a_k = softmax([log(alphas) + G] / lambda)_k
                '''
                logits = self.ac.actor(observations) # (B x A)
                alphas = -F.log_softmax(logits, dim=1) # (B x A)

                gumbel_noises = Gumbel(0, 1).sample(alphas.shape) # (B x A)
                noised_alphas = torch.log(alphas) + gumbel_noises # (B x A)

                actions_gm = torch.argmax(noised_alphas, dim=1) # (B, )
                # Convert to one-hot to have the shape (B x A)
                actions_gm = F.one_hot(actions_gm, num_classes=self.action_dim) # (B x A)
                actions_gs = F.softmax(noised_alphas / self.lambd, dim=1) # (B x A)

                actions_ = (actions_gm - actions_gs).detach() + actions_gs # (B x A)
                # Convert from one-hot to normal form of action
                actions_ = torch.argmax(actions_, dim=1) # (B, )

            loss = -self.ac.critic(observations, actions_).mean()
            return loss

        def compute_targets(rewards, next_observations, terminated):
            if self.is_continuous:
                actions_target = self.ac_target.actor(next_observations)
            else:
                logits_target = self.ac_target.actor(next_observations)
                actions_target_dist = Categorical(logits=logits_target)
                actions_target = actions_target_dist.sample()
            return rewards + self.gamma * (1 - terminated) \
                * self.ac_target.critic(next_observations, actions_target)

        def compute_q_loss(observations, actions, targets):
            q_values = self.ac.critic(observations, actions)
            loss = ((q_values - targets) ** 2).mean()
            return loss, q_values

        # (B x O), (B x A) or (B x 1), (B x 1), (B x O), (B x 1)
        observations, actions, rewards, next_observations, terminated \
            = map(lambda x: x.to(self.device), self.buffer.get(self.batch_size))

        targets = compute_targets(rewards, next_observations, terminated)

        self.critic_opt.zero_grad()
        q_loss, q_values = compute_q_loss(observations, actions, targets)
        q_loss.backward()
        self.critic_opt.step()
        
        self.actor_opt.zero_grad()
        mu_loss = compute_mu_loss(observations)
        mu_loss.backward()
        self.actor_opt.step()

        # Update target networks parameters according to Polyak averaging
        soft_update(self.ac, self.ac_target, self.tau)
        self.logger.add({
            'mu-loss': mu_loss.item(),
            'q-loss': q_loss.item(),
            'q-values': q_values.detach().cpu().numpy()
        })


    def select_action(self, observation: np.ndarray, deterministic=True, noise_sigma: float=0.0) -> np.ndarray:
        observation = to_tensor(observation, self.device)
        with torch.no_grad():
            if self.is_continuous:
                epsilon = noise_sigma * np.random.randn(self.action_dim)
                action = self.ac.actor(observation).cpu().numpy() + epsilon
                return np.clip(action, -self.action_limit, self.action_limit)
            else:
                logits = self.ac.actor(observation) # (A, )
                if deterministic:
                    return torch.argmax(logits).cpu().numpy()
                action_dist = Categorical(logits=logits)
                action = action_dist.sample() # (A, )
                return action.cpu().numpy()


    def load(self, model_path: str) -> None:
        """Model loading"""
        self.ac.load_state_dict(torch.load(model_path))


    def test(self) -> None:
        env = deepcopy(self.env)
        for _ in range(self.test_episodes):
            observation, _ = env.reset()
            rewards = []
            while True:
                action = self.select_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)

                if terminated or len(rewards) == self.max_ep_len:
                    self.logger.add({
                        'test-episode-return': sum(rewards),
                        'test-episode-length': len(rewards)
                    })
                    break


    def train(self) -> None:
        step = 0
        for epoch in range(1, self.epochs + 1):

            while True:
                observation, _ = self.env.reset()
                rewards = []

                while True:
                    if (epoch - 1) * self.steps_per_epoch + step <= self.start_step:
                        # SpinniningUP's trick to ultilize exploration at the beginning
                        action = self.env.action_space.sample()
                    else:
                        action = self.select_action(observation, deterministic=False, noise_sigma=self.sigma)
                    next_observation, reward, terminated, truncated, _ = self.env.step(action)
                    rewards.append(reward)
                    step += 1

                    # Set `terminated` to `False` in case episode is forced to stopped by the env
                    terminated = False if len(rewards) == self.max_ep_len else terminated
                    self.buffer.add(observation, action, reward, next_observation, terminated)
                    observation = next_observation

                    if step >= self.update_after and step % self.update_every == 0:
                        for _ in range(self.update_every):
                            self.update_params()

                    if terminated or len(rewards) == self.max_ep_len or step % self.steps_per_epoch == 0:
                        self.logger.add({
                            'episode-return': sum(rewards),
                            'episode-length': len(rewards)
                        })
                        break
                if step % self.steps_per_epoch == 0:
                    if self.save and epoch % self.save_every == 0:
                        self.logger.save_state()

                    self.test()

                    self.logger.log_epoch('epoch', epoch)
                    self.logger.log_epoch('mu-loss', average_only=True)
                    self.logger.log_epoch('q-loss', average_only=True)
                    self.logger.log_epoch('q-values', need_optima=True)
                    self.logger.log_epoch('episode-return', need_optima=True)
                    self.logger.log_epoch('episode-length', average_only=True)
                    self.logger.log_epoch('test-episode-return', need_optima=True)
                    self.logger.log_epoch('test-episode-length', average_only=True)
                    self.logger.log_epoch('total-env-interacts', step)
                    self.logger.dump_epoch()
                    break
        self.env.close()
        if self.render:
            self.logger.render(self.select_action)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradient')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='ddpg',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256, 256],
                        help="Sizes of policy & value function networks' hidden layers")
    parser.add_argument('--mu-lr', type=float, default=1e-3,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--q-lr', type=float, default=1e-3,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--lambd', type=float, default=1.0,
                        help='Softmax temperature parameter, used in Gumbel-Softmax trick')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Minibatch size')
    parser.add_argument('--start-step', type=int, default=10000,
                        help='Start step to begin action selection according to policy network')
    parser.add_argument('--update-every', type=int, default=50,
                        help='Parameters update frequency')
    parser.add_argument('--update-after', type=int, default=1000,
                        help='Number of steps after which update is allowed')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft (Polyak averaging) update coefficient')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Standard deviation of mean-zero Gaussian noise for exploration')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of episodes to test the deterministic policy at the end of each epoch')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the final model')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the training result')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the training statistics')
    args = parser.parse_args()

    agent = DDPG(args)
    agent.train()
