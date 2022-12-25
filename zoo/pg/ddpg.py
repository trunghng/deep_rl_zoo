import gym
import torch
import numpy as np
from torch.optim import Adam
from torch.autograd import grad
import argparse, random, os
from copy import deepcopy
from common.logger import Logger
from zoo.pg.network import MLPDeterministicActorCritic
from zoo.pg.utils import ReplayBuffer


class DDPG:


    def __init__(self, args):
        '''
        Deep Determinisitic Policy Gradient

        :param env: (str) OpenAI environment name
        :param seed: (int) Seed for RNG
        :param hidden_layers: (List[int]) Hidden layers size of policy & Q networks
        :param pi_lr: (float) Learning rate for policy optimizer
        :param q_lr: (float) Learning rate for value function optimizer
        :param epochs: (int) Number of epochs
        :param steps_per_epoch: (int) Maximum number of steps per epoch
        :param max_ep_len: (int) Maximum length of an episode
        :param buffer_size: (int) Replay buffer size
        :param batch_size: (int) Minibatch size
        :param start_step: (int) Start step to begin select action according to policy network
        :param update_every: (int) Parameters update frequency
        :param update_after: (int) Number of steps after which paramters update is allowed. 
                    This guarantees there are enough number of training experience in the replay buffer
        :param gamma: (float) Discount factor
        :param rho: (float) Hyperparameter for polyak averaging
        :param sigma: (float) Standard deviation of mean-zero Gaussian noise for exploration.
                    The original DDPG used Ornstein-Uhlenbeck process instead.
        :param goal: (float) Total reward threshold for early stopping
        :param save: (bool) Whether to save the final model
        :param save_freq: (int) Model saving frequency
        :param render: (bool) Whether to render the training result in video
        :param plot: (bool) Whether to plot the statistics and save as image
        :param model_dir: (str) Model directory
        :param video_dir: (str) Video directory
        :param figure_dir: (str) Figure directory
        '''
        self.env = gym.make(args.env)
        self.seed(args.seed)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.ac = MLPDeterministicActorCritic(observation_space, action_space, args.hidden_layers)
        self.ac_target = deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False
        self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.pi_lr)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.q_lr)
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.max_ep_len = args.max_ep_len
        self.buffer = ReplayBuffer(args.buffer_size, observation_space.shape, action_space.shape[0])
        self.batch_size = args.batch_size
        self.start_step = args.start_step
        self.update_every = args.update_every
        self.update_after = args.update_after
        self.gamma = args.gamma
        self.rho = args.rho
        self.sigma = args.sigma
        self.goal = args.goal
        self.save = args.save
        self.save_freq = args.save_freq
        self.render = args.render
        self.plot = args.plot
        self.logger = Logger(args.env, args.model_dir, args.video_dir, args.figure_dir)
        self.logger.log(f'Algorithm: DDPG\nEnvironment: {args.env}\nSeed: {args.seed}')
        self.logger.set_saver(self.ac)


    def seed(self, seed):
        '''
        Set global seed
        '''
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)


    def update_params(self):
        '''
        Update policy and value networks' parameters
        '''
        def compute_targets(rewards, next_observations, terminated):
            next_actions = self.ac_target.actor(next_observations)
            return rewards + self.gamma * (1 - terminated) * self.ac_target.critic(next_observations, next_actions)

        def compute_q_loss(observations, actions, targets):
            q_values = self.ac.critic(observations, actions)
            loss = ((q_values - targets) ** 2).mean()
            return loss, q_values

        def compute_pi_loss(observations):
            loss = -self.ac.critic(observations, self.ac.actor(observations)).mean()
            return loss

        def update_targets():
            with torch.no_grad():
                for p_target, p in zip(self.ac_target.parameters(), self.ac.parameters()):
                    p_target.data.copy_(p_target.data * self.rho + (1 - self.rho) * p.data)

        observations, actions, rewards, next_observations, terminated = self.buffer.sample(self.batch_size)
        targets = compute_targets(rewards, next_observations, terminated)

        self.critic_opt.zero_grad()
        q_loss, q_values = compute_q_loss(observations, actions, targets)
        q_loss.backward()
        self.critic_opt.step()
        
        self.actor_opt.zero_grad()
        pi_loss = compute_pi_loss(observations)
        pi_loss.backward()
        self.actor_opt.step()

        update_targets()
        self.logger.add(PiLoss=pi_loss.item(), QLoss=q_loss.item(), QValues=q_values.detach().numpy())


    def train(self):
        step = 0
        for epoch in range(1, self.epochs + 1):

            while True:
                observation = self.env.reset()
                rewards = []

                while True:
                    action = self.ac.act(observation, self.sigma)
                    next_observation, reward, terminated, _ = self.env.step(action)
                    rewards.append(reward)
                    step += 1

                    terminated = False if len(rewards) == self.max_ep_len else terminated
                    self.buffer.add(observation, action, reward, next_observation, terminated)
                    observation = next_observation

                    if step >= self.update_after and step % self.update_every == 0:
                        for _ in range(self.update_every):
                            self.update_params()

                    if terminated or len(rewards) == self.max_ep_len or step % self.steps_per_epoch == 0:
                        self.logger.add(Return=sum(rewards), EpLen=len(rewards))
                        break
                if step % self.steps_per_epoch == 0:
                    break

            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('PiLoss')
            self.logger.log_tabular('QLoss')
            self.logger.log_tabular('QValues')
            self.logger.log_tabular('Return')
            self.logger.log_tabular('EpLen')
            self.logger.log()

            if self.save and epoch % self.save_freq == 0:
                self.logger.save_state()
        self.env.close()
        if self.render:
            self.logger.render(self.env)
        if self.plot:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'HalfCheetah-v2'],
                        help='Environment ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-layers', nargs='+', type=int,
                        help='Hidden layers size of policy & value function networks')
    parser.add_argument('--pi-lr', type=float, default=1e-3,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--q-lr', type=float, default=1e-3,
                        help='Learning rate for value function optimizer')
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
    parser.add_argument('--rho', type=float, default=0.995,
                        help='Hyperparameter for polyak averaging')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Standard deviation of mean-zero Gaussian noise for exploration')
    parser.add_argument('--goal', type=int,
                        help='Total reward threshold for early stopping')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    parser.add_argument('--model-dir', type=str, default='./zoo/pg/output/models/ddpg',
                        help='Where to save the model')
    parser.add_argument('--video-dir', type=str, default='./zoo/pg/output/videos/ddpg',
                        help='Where to save the video output')
    parser.add_argument('--figure-dir', type=str, default='./zoo/pg/output/figures/ddpg',
                        help='Where to save the plots')
    args = parser.parse_args()

    agent = DDPG(args)
    agent.train()
