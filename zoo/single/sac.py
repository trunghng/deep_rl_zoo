from typing import List
import argparse, random, os, itertools
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Space, Box
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import grad

from common.utils import to_tensor, soft_update, dim, set_seed
from common.policy import SoftPolicy
from common.vf import StateActionValueFunction
from common.buffer import ReplayBuffer
from common.logger import Logger


class ActorCritic(nn.Module):

    def __init__(self,
                observation_space: Space,
                action_space: Space,
                hidden_sizes: List[int]=[256, 256],
                activation: nn.Module=nn.ReLU,
                device: str='cpu') -> None:
        super().__init__()
        obs_dim = dim(observation_space)
        action_dim = dim(action_space)
        action_limit = action_space.high[0]
        self.pi = SoftPolicy(obs_dim, action_dim, hidden_sizes, activation, action_limit).to(device)
        self.q1 = StateActionValueFunction(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.q2 = deepcopy(self.q1)


class SAC:
    """Soft Actor-Critic

    :param env: (str) Environment ID.
    :param exp_name: (str) Experiment name.
    :param seed: (int) Seed for RNG.
    :param hidden_sizes: (List[int]) Sizes of policy & Q networks' hidden layers
    :param lr: (float) Learning rate for policy, Q networks & entropy coefficient optimizers.
    :param epochs: (int) Number of epochs to train agent.
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
    :param ent_coeff: (float) Entropy regularization coefficient.
    :param ent_coeff_init: (float) Initial value for automating entropy adjustment scheme.
    :param ent_target: (float) Desired entropy, used for automating entropy adjustment.
    :param adjust_ent_coeff: (bool) Whether to use automating entropy adjustment scheme, 
                use fixed `ent_coeff` otherwise.
    :param test_episodes: (int) Number of episodes to test the deterministic policy at the end of each episode.
    :param save: (bool) Whether to save the final model.
    :param save_every: (int) Model saving frequency.
    :param render: (bool) Whether to render the training result.
    :param plot: (bool) Whether to plot the training statistics.
    """

    def __init__(self, args) -> None:
        algo = 'sac'
        self.env = gym.make(args.env)
        set_seed(args.seed)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        assert isinstance(action_space, Box), f'{algo} does not work with discrete action space env!'
        self.ac = ActorCritic(observation_space, action_space, args.hidden_sizes, device=self.device)
        self.ac_target = deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False
        self.pi_opt = Adam(self.ac.pi.parameters(), lr=args.lr)
        self.q_opt = Adam(itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()), lr=args.lr)
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
        if args.adjust_ent_coeff:
            # Set entropy_target = -dim(A) if not specified
            self.ent_target = -dim(action_space) if args.ent_target == 'auto' else float(args.ent_target)

            # Optimize log(alpha) instead according to stable-baseline3
            assert args.ent_coeff_init > 0, 'Initial value for entropy temperature must be greater than 0!'
            self.log_ent_coeff = torch.log(torch.tensor(args.ent_coeff_init, device=self.device)).requires_grad_(True)
            self.ent_coeff_opt = Adam([self.log_ent_coeff], lr=args.lr)
        else:
            self.ent_coeff = args.ent_coeff
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
        config_dict = vars(args)
        config_dict['algo'] = algo
        self.logger = Logger(log_dir=log_dir)
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.ac)


    def update_params(self) -> None:
        """Update policy & Q networks' parameters"""

        def compute_targets(rewards, next_observations, terminated):
            """Compute TD targets for Q functions"""
            with torch.no_grad():
                next_actions, logp_next_actions = self.ac.pi(next_observations)

                q1_target = self.ac_target.q1(next_observations, next_actions)
                q2_target = self.ac_target.q2(next_observations, next_actions)
                q_target = torch.min(q1_target, q2_target)
                targets = rewards + self.gamma * (1 - terminated) * (q_target - self.ent_coeff * logp_next_actions)
            return targets

        def compute_q_values(observations, actions):
            q1_values = self.ac.q1(observations, actions)
            q2_values = self.ac.q2(observations, actions)
            return q1_values, q2_values

        def compute_q_loss(observations, actions, targets):
            q1_values, q2_values = compute_q_values(observations, actions)
            q1_loss = ((targets - q1_values) ** 2).mean()
            q2_loss = ((targets - q2_values) ** 2).mean()
            q_loss = q1_loss + q2_loss
            return q_loss, q1_values, q2_values

        def compute_pi_loss(observations, actions, logp_actions):
            q1_values, q2_values = compute_q_values(observations, actions)
            q_values = torch.min(q1_values, q2_values)
            pi_loss = (self.ent_coeff * logp_actions - q_values).mean()
            return pi_loss

        observations, actions, rewards, next_observations, terminated \
            = map(lambda x: x.to(self.device), self.buffer.get(self.batch_size))

        # Sample actions with current pi
        actions_, logp_actions = self.ac.pi(observations)

        # Whether to enable entropy temperature adjustment
        if hasattr(self, 'ent_target'):
            self.ent_coeff = torch.exp(self.log_ent_coeff.detach()).item()
            self.ent_coeff_opt.zero_grad()
            ent_coeff_loss = -(self.log_ent_coeff * (logp_actions + self.ent_target).detach()).mean()
            ent_coeff_loss.backward()
            self.ent_coeff_opt.step()
            self.logger.add({'entropy-coeff-loss': ent_coeff_loss.item()})

        self.q_opt.zero_grad()
        targets = compute_targets(rewards, next_observations, terminated)
        q_loss, q1_values, q2_values = compute_q_loss(observations, actions, targets)
        q_loss.backward()
        self.q_opt.step()

        self.pi_opt.zero_grad()
        pi_loss = compute_pi_loss(observations, actions_, logp_actions)
        pi_loss.backward()
        self.pi_opt.step()

        # Update target networks parameters according to Polyak average
        soft_update(self.ac, self.ac_target, self.tau)
        self.logger.add({
            'pi-loss': pi_loss.item(),
            'q-loss': q_loss.item(),
            'q1-values': q1_values.detach().cpu().numpy(),
            'q2-values': q2_values.detach().cpu().numpy(),
            'log-probs': logp_actions.detach().cpu().numpy(),
            'entropy-coeff': self.ent_coeff
        })


    def select_action(self, observation: np.ndarray, deterministic: bool=True) -> np.ndarray:
        observation = to_tensor(observation, self.device)
        with torch.no_grad():
            action, _ = self.ac.pi(observation, deterministic, need_logprob=False)
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

                if terminated or truncated or len(rewards) == self.max_ep_len:
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
                        action = self.select_action(observation, deterministic=False)
                    next_observation, reward, terminated, _, _ = self.env.step(action)

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
                    self.logger.log_epoch('pi-loss', average_only=True)
                    self.logger.log_epoch('q-loss', average_only=True)
                    self.logger.log_epoch('q1-values', need_optima=True)
                    self.logger.log_epoch('q2-values', need_optima=True)
                    self.logger.log_epoch('episode-return', need_optima=True)
                    self.logger.log_epoch('episode-length', average_only=True)
                    self.logger.log_epoch('test-episode-return', need_optima=True)
                    self.logger.log_epoch('test-episode-length', average_only=True)
                    self.logger.log_epoch('entropy-coeff', average_only=True)
                    self.logger.log_epoch('total-env-interacts', step)
                    if hasattr(self, 'ent_target'):
                        self.logger.log_epoch('entropy-coeff-loss', average_only=True)
                    self.logger.dump_epoch()
                    break
        self.env.close()
        if self.render:
            self.logger.render(self.select_action)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soft Actor-Critic')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='sac',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256, 256],
                        help="Sizes of policy & Q networks' hidden layers")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for policy, Q networks & entropy coefficient optimizers')
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
    parser.add_argument('--ent-coeff', type=float, default=0.2,
                        help='Entropy regularization coefficient')
    parser.add_argument('--adjust-ent-coeff', action='store_false',
                        help='Whether to enable automating entropy adjustment scheme')
    parser.add_argument('--ent-coeff-init', type=float, default=1.0, 
                        help='Initial value for automating entropy adjustment scheme')
    parser.add_argument('--ent-target', type=str, default='auto',
                        help='Desired entropy, used for automating entropy adjustment')
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

    agent = SAC(args)
    agent.train()
