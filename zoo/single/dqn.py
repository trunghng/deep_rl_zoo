from typing import List, Tuple
import argparse, sys, random, os
import os.path as osp
from copy import deepcopy

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from common.vf import MLPDeepQNet, CNNDeepQNet, MLPDuelingQNet
from common.utils import set_seed, make_atari_env, soft_update, hard_update, dim, to_tensor
from common.buffer import ReplayBuffer
from common.logger import Logger


class DQN:
    """
    Deep-Q Learning

    :param env: (str) Environment ID
    :param exp_name: (str) Experiment name.
    :param seed: (int) Seed for RNG.
    :param atari: (bool) Whether to use atari environment
    :param double_q: (bool) Whether to use double Q-learning
    :param dueling_net: (bool) Whether to use dueling Q-network
    :param epochs: (int) Number of epochs.
    :param steps_per_epoch: (int) Maximum number of steps per epoch.
    :param max_ep_len: (int) Maximum length of an episode.
    :param soft_update: (bool) Whether to use soft update.
    :param tau: (float) Soft (Polyak averaging) update coefficient
    :param epsilon_init: (float) Initial value for epsilon linearly annealing
    :param epsilon_final: (float) Final value for epsilon linearly annealing
    :param epsilon_decay: (float) Decay value for epsilon linearly annealing
    :param gamma: (float) Discount factor
    :param lr: (float) Learning rate for Q-network optimizer
    :param buffer_size: (int) Replay buffer size
    :param batch_size: (int) Minibatch size
    :param train_every: (int) Number of actions selected by the agent between successive SGD updates.
    :param update_every: (int) Target network update frequency.
    :param test_episodes: (int) Number of episodes to test the deterministic policy at the end of each epoch.
    :param save: (bool) Whether to save the final model.
    :param save_every: (int) Model saving frequency.
    :param render: (bool) Whether to render the training result.
    :param plot: (bool) Whether to plot the training statistics.
    """

    def __init__(self, args) -> None:
        self.env = make_atari_env(args.env) if args.atari else gym.make(args.env)
        obs_dim = dim(self.env.observation_space)
        action_dim = dim(self.env.action_space)
        action_dim = self.env.action_space.shape[0] if isinstance(self.env.action_space, Box) \
            else self.env.action_space.n

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        set_seed(args.seed)
        if args.atari:
            network = CNNDeepQNet
        elif args.dueling_net:
            network = MLPDuelingQNet
        else:
            network = MLPDeepQNet
        self.Qnet = network(obs_dim, action_dim).to(self.device)
        self.Qnet_target = deepcopy(self.Qnet)
        for p in self.Qnet_target.parameters():
            p.requires_grad = False
        if args.double_q:
            self.compute_td_error = self.compute_td_loss_double_q
        else:
            self.compute_td_error = self.compute_td_loss
        self.optimizer = opt.Adam(self.Qnet.parameters(), lr=args.lr)
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.max_ep_len = args.max_ep_len
        self.soft_update = args.soft_update
        self.tau = args.tau
        self.epsilon_init = args.epsilon_init
        self.epsilon_final = args.epsilon_final
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma
        self.buffer = ReplayBuffer(args.buffer_size)
        self.batch_size = args.batch_size
        self.train_every = args.train_every
        self.update_every = args.update_every
        self.test_episodes = args.test_episodes
        self.save = args.save
        self.save_every = args.save_every
        self.render = args.render
        self.plot = args.plot
        self.step = 0
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        config_dict = vars(args)
        config_dict['algo'] = 'dqn'
        self.logger = Logger(log_dir=log_dir)
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.Qnet)


    def anneal_epsilon(self, ep: int) -> float:
        """Epsilon linearly annealing

        :param ep: current episode
        :return epsilon: epsilon for greedy-epsilon
        """
        return self.epsilon_final + (self.epsilon_init - self.epsilon_final) \
                * np.exp(-1. * ep / self.epsilon_decay)


    def select_action(self, observation: np.ndarray, epsilon: float=0.0) -> int:
        """
        Select action according to the behavior policy
            here we use epsilon-greedy as behavior policy
        """
        if random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            observation = to_tensor(observation, self.device)
            with torch.no_grad():
                action_values = self.Qnet(observation.float())
            action = torch.argmax(action_values).item()
        return action


    def compute_td_loss(self, observations, actions, rewards, next_observations, terminated):
        """Compute TD error error according to Q-learning"""
        q_target_next = torch.max(self.Qnet_target(next_observations), dim=1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_target_next * (1 - terminated)
        q_expected = self.Qnet(observations).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss, q_expected


    def compute_td_loss_double_q(self, observations, actions, rewards, next_observations, terminated):
        """Compute TD error according to double Q-learning"""
        greedy_actions = torch.argmax(self.Qnet(next_observations), dim=1).unsqueeze(1)
        q_target_greedy = self.Qnet_target(next_observations).gather(1, greedy_actions)
        q_target = rewards.view(-1, 1) + self.gamma * q_target_greedy * (1 - terminated.view(-1, 1))
        q_expected = self.Qnet(observations).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss, q_expected


    def update_target_params(self) -> None:
        """Update target network's parameters"""
        if self.soft_update:
            soft_update(self.Qnet, self.Qnet_target, self.tau)
        else:
            hard_update(self.Qnet, self.Qnet_target)


    def learn(self) -> None:
        if self.step % self.train_every == 0 and len(self.buffer) >= self.batch_size:
            observations, actions, rewards, next_observations, terminated = self.buffer.get(self.batch_size)
            observations = observations.float().to(self.device)
            actions = actions.long().to(self.device)
            rewards = rewards.float().to(self.device)
            next_observations = next_observations.float().to(self.device)
            terminated = terminated.float().to(self.device)

            loss, value = self.compute_td_error(observations, actions, rewards, next_observations, terminated)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.add({
                'loss': loss.item(),
                'values': value.detach().cpu().numpy()
            })

        if self.step % self.update_every == 0:
            self.update_target_params()
        self.step += 1


    def load(self, model_path: str) -> None:
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

                if terminated or truncated or len(rewards) == self.max_ep_len:
                    self.logger.add({
                        'test-episode-return': sum(rewards),
                        'test-episode-length': len(rewards)
                    })
                    break


    def train(self) -> None:
        total_env_interacts = 0
        ep = 0

        for epoch in range(1, self.epochs + 1):

            while True:
                epsilon = self.anneal_epsilon(ep)
                observation, _ = self.env.reset()
                rewards = []

                while True:
                    action = self.select_action(observation, epsilon)
                    next_observation, reward, terminated, truncated, _ = self.env.step(action)
                    rewards.append(reward)

                    self.buffer.add(observation, action, reward, next_observation, terminated)
                    self.learn()
                    observation = next_observation

                    if terminated or truncated or len(rewards) == self.max_ep_len or self.step % self.steps_per_epoch == 0:
                        self.logger.add({
                            'episode-return': sum(rewards),
                            'episode-length': len(rewards)
                        })
                        ep += 1
                        break

                if self.step % self.steps_per_epoch == 0:
                    if self.save and epoch % self.save_every == 0:
                        self.logger.save_state()
                    self.test()

                    self.logger.log_epoch('epoch', epoch)
                    self.logger.log_epoch('loss', average_only=True)
                    self.logger.log_epoch('values', need_optima=True)
                    self.logger.log_epoch('episode-return', need_optima=True)
                    self.logger.log_epoch('episode-length', average_only=True)
                    self.logger.log_epoch('test-episode-return', need_optima=True)
                    self.logger.log_epoch('test-episode-length', average_only=True)
                    self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
                    self.logger.dump_epoch()
                    break
        self.env.close()
        if self.render:
            self.logger.render(self.select_action)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')

    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='OpenAI enviroment name')
    parser.add_argument('--exp-name', type=str, default='dqn',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--atari', action='store_true',
                        help='Whether to use atari environment')
    parser.add_argument('--double-q', action='store_true',
                        help='Whether to use double Q-learning')
    parser.add_argument('--dueling-net', action='store_true',
                        help='Whether to use dueling Q-network')
    parser.add_argument('--epochs', type=int, default=100,
                        help = 'Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--soft-update', action='store_true',
                        help='Whether to use soft update')
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='Soft (Polyak averaging) update coefficient')
    parser.add_argument('--epsilon-init', type=float, default=1,
                        help='Initial value of epsilon')
    parser.add_argument('--epsilon-final', type=float, default=1e-2,
                        help='Final value of epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=200,
                        help='Final value of epsilon')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--train-every', type=int, default=4,
                        help='Number of steps between optimization steps')
    parser.add_argument('--update-every', type=int, default=4,
                        help='Target network update frequency')
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

    agent = DQN(args)
    agent.train()
