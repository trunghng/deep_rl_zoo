import argparse, os
from typing import List, Dict, Tuple
from collections import defaultdict
from copy import deepcopy

from gymnasium.spaces import Space, Box, Discrete
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from common.policy import DeterministicPolicy, DiscretePolicy
from common.vf import MADDPGStateActionValueFunction
from common.utils import dim, make_mpe_env, set_seed, to_tensor, split_obs_action, soft_update
from common.buffer import ReplayBuffer
from common.logger import Logger


class ActorCritic(nn.Module):


    def __init__(self,
                agent: str,
                obs_space_dict: Dict[str, Space],
                action_space_dict: Dict[str, Space],
                hidden_sizes: List[int]=[64, 64],
                activation=nn.ReLU,
                device: str='cpu') -> None:
        super().__init__()
        obs_dim = dim(obs_space_dict[agent])
        obs_dims = list(map(lambda aid: dim(obs_space_dict[aid]), obs_space_dict))
        action_dim = dim(action_space_dict[agent])
        action_dims = list(map(lambda aid: dim(action_space_dict[aid]), action_space_dict))

        action_space = action_space_dict[agent]
        # continuous action
        if isinstance(action_space, Box):
            action_limit = action_space.high[0]
            self.mu = DeterministicPolicy(obs_dim, action_dim, hidden_sizes, 
                            activation, nn.Sigmoid, action_limit).to(device)
        # discrete action
        elif isinstance(action_space, Discrete):
            self.mu = DiscretePolicy(obs_dim, action_dim, hidden_sizes, activation, nn.Indentiy).to(device)
        self.mu_target = deepcopy(self.mu)
        for p in self.mu_target.parameters():
            p.requires_grad = False
        self.q = MADDPGStateActionValueFunction(obs_dims, action_dims, 
                    hidden_sizes, activation).to(device)


class MADDPG:
    """Multi-agent DDPG"""

    def __init__(self, args) -> None:
        set_seed(args.seed)
        self.env = make_mpe_env(args.env, continuous_actions=True)
        self.env.reset()
        self.ac, self.mu_opt, self.q_opt, self.obs_dim, self.action_dim, self.is_continuous\
            = dict(), dict(), dict(), dict(), dict(), dict()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for agent in self.env.agents:
            self.ac[agent] = ActorCritic(agent, self.env.observation_spaces, 
                                    self.env.action_spaces, args.hidden_sizes, device=self.device)
            self.mu_opt[agent] = Adam(self.ac[agent].mu.parameters(), lr=args.lr)
            self.q_opt[agent] = Adam(self.ac[agent].q.parameters(), lr=args.lr)
            self.obs_dim[agent] = dim(self.env.observation_spaces[agent])
            self.action_dim[agent] = dim(self.env.action_spaces[agent])
            self.is_continuous[agent] = isinstance(self.env.action_spaces[agent], Box)

        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.max_ep_len = args.max_ep_len
        self.buffer = ReplayBuffer(args.buffer_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.update_every = args.update_every
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
        config_dict['algo'] = 'maddpg'
        self.logger.save_config(config_dict)
        self.logger.set_saver(list(map(lambda agent: self.ac[agent], self.ac)))


    def get_info(self, env) -> Tuple[np.ndarray, List[np.ndarray], List[bool], List[bool]]:
        rewards, terminations, truncations = [], [], []
        for agent in env.agents:
            rewards.append(env.rewards[agent])
            terminations.append(env.terminations[agent])
            truncations.append(env.truncations[agent])
        return env.state(), rewards, terminations, truncations


    def update_ac_params(self) -> None:
        """Update actor-critic's parameters"""
        mu_losses, q_losses, q_values_ = [], [], []

        for i, agent in enumerate(self.env.agents):

            def compute_targets(observations, rewards, next_observations, terminations):
                """Compute TD targets y for Q functions"""
                # O_1, ..., O_N
                obs_dims = [self.obs_dim[agent_] for agent_ in self.obs_dim]
                observations_ = split_obs_action(observations, obs_dims) # (B x O_1), ..., (B x O_N)

                '''
                Compute a_1', ..., a_N':
                    a_k' = (mu_target)_k(o_k)
                (B x A_1), ..., (B x A_N)
                '''
                next_actions = [self.ac[agent_].mu_target(obs) for agent_, obs in zip(self.ac, observations_)]
                next_actions = torch.cat(next_actions, dim=1) # (B x joint_A)

                # Q_i(o_1', ..., o_N', a_1', ..., a_N')
                q_target_values = self.ac[agent].q(next_observations, next_actions) # (B x 1)

                '''
                Compute targets:
                    y = r_i + gamma * (1 - d_i) * Q_i(o_1', ..., o_N', a_1', ..., a_N')
                '''
                y = rewards[:, i].view(self.batch_size, 1) + self.gamma * \
                        (1 - terminations[:, i].view(self.batch_size, 1)) * q_target_values # (B x 1)
                return y

            def compute_critic_loss(observations, actions, rewards, next_observations, terminations):
                """Compute critic loss"""
                y = compute_targets(observations, rewards, next_observations, terminations) # (B x 1)
                q_values = self.ac[agent].q(observations, actions) # (B x 1)
                q_loss = ((y - q_values) ** 2).mean() # (1,)
                return q_loss, q_values

            def compute_actor_loss(observations, actions):
                """Compute actor loss"""
                # O_1, ..., O_N
                obs_dims = [self.obs_dim[agent_] for agent_ in self.obs_dim]
                observation = split_obs_action(observations, obs_dims)[i] # (B x O_i)

                # A_1, ..., A_N
                action_dims = [self.action_dim[agent_] for agent_ in self.action_dim]
                actions_ = list(split_obs_action(actions, action_dims)) # (B x A_1), ..., (B x A_N)
                actions_[i] = self.ac[agent].mu(observation).float() # (B x A_i)
                actions_ = torch.cat(actions_, dim=1) # (B x joint_A)

                mu_loss = -self.ac[agent].q(observations, actions_).mean() # (1,)
                return mu_loss

            # (B x joint_O), (B x joint_A), (B x N), (B x joint_O), (B x N)
            observations, actions, rewards, next_observations, terminations \
                = map(lambda x: x.to(self.device), self.buffer.get(self.batch_size))
                
            self.q_opt[agent].zero_grad()
            q_loss, q_values = compute_critic_loss(observations, actions, rewards, next_observations, terminations)
            q_loss.backward()
            self.q_opt[agent].step()

            self.mu_opt[agent].zero_grad()
            mu_loss = compute_actor_loss(observations, actions)
            mu_loss.backward()
            self.mu_opt[agent].step()

            q_losses.append(q_loss.item())
            q_values_.append(q_values.detach().cpu().numpy())
            mu_losses.append(mu_loss.item())

        self.logger.add({
            'mu-loss': np.asarray(mu_losses).mean(),
            'q-loss': np.asarray(q_losses).mean(),
            'q-values': np.asarray(q_values_).mean()
        })


    def update_target_params(self) -> None:
        """Update target network's parameters"""
        for agent in self.env.agents:
            soft_update(self.ac[agent].mu, self.ac[agent].mu_target, self.tau)


    def select_action(self, agent: str, observation: torch.Tensor, noise_sigma: float=0.0) -> np.ndarray:
        """
        :param agent: agent ID
        :param observation: current observation
        :param noise_sigma: Standard deviation of mean-zero Gaussian noise for exploration
        """
        observation = to_tensor(observation, self.device)
        with torch.no_grad():
            if isinstance(self.env.action_space(agent), Box):
                epsilon = noise_sigma * np.random.randn(self.action_dim[agent])
                action = self.ac[agent].mu(observation).cpu().numpy() + epsilon
                action = np.clip(action, self.env.action_space(agent).low, 
                            self.env.action_space(agent).high, dtype=np.float32)
            elif isinstance(self.env.action_space(agent), Discrete):
                pass
        return action


    def test(self) -> None:
        env = deepcopy(self.env)
        for _ in range(self.test_episodes):
            env.reset()
            total_rewards = []

            while True:
                for agent in env.agents:
                    observation = env.observe(agent)
                    action = self.select_action(agent, observation)
                    env.step(action)

                _, rewards, terminations, truncations = self.get_info(env)
                total_rewards.append(sum(rewards))

                if all(terminations) or all(truncations) or len(total_rewards) == self.max_ep_len:
                    self.logger.add({
                        'test-episode-return': sum(total_rewards),
                        'test-episode-length': len(total_rewards)
                    })
                    break


    def train(self) -> None:
        step = 0
        for epoch in range(1, self.epochs + 1):

            while True:
                self.env.reset()
                total_rewards = []

                while True:
                    actions = []
                    for agent in self.env.agents:
                        observations = self.env.state()
                        action = self.select_action(agent, self.env.observe(agent), self.sigma)
                        self.env.step(action)
                        actions.append(action)
                    actions = np.concatenate(actions)
                    next_observations, rewards, terminations, truncations = self.get_info(self.env)
                    self.buffer.add(observations, actions, rewards, next_observations, terminations)
                    step += 1
                    total_rewards.append(sum(rewards))

                    if len(self.buffer) >= self.batch_size:
                        self.update_ac_params()

                    if step % self.update_every == 0:
                        self.update_target_params()

                    if all(terminations) or all(truncations) or step % self.steps_per_epoch == 0 \
                            or len(total_rewards) == self.max_ep_len:
                        self.logger.add({
                            'episode-return': sum(total_rewards),
                            'episode-length': len(total_rewards)
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
                    self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
                    self.logger.dump_epoch()
                    break

        self.env.close()
        if self.render:
            self.logger.render(self.selection_action)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')

    parser.add_argument('--env', type=str, choices=['adversary', 'crypto', 'push',
                        'reference', 'speaker-listener', 'spread', 'tag', 'world-comm',
                        'simple'], default='adversary', help='PettingZoo MPE environment')
    parser.add_argument('--exp-name', type=str, default='mpe',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 64],
                        help="Sizes of policy & value function networks' hidden layers.")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--max-ep-len', type=int, default=25,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Minibatch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for policy & Q network optimizer')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=1e-2,
                        help='Smoothness parameter, used for target network soft update')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Standard deviation of mean-zero Gaussian noise for exploration')
    parser.add_argument('--update-every', type=int, default=100,
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

    model = MADDPG(args)
    model.train()
