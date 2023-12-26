import argparse, os
from typing import List, Dict, Tuple
from collections import defaultdict
from copy import deepcopy

from gymnasium.spaces import Space, Box, Discrete
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from common.policy import DeterministicPolicy, MADDPGPolicy
from common.vf import MADDPGStateActionValueFunction
from common.utils import dim, make_mpe_env, set_seed, to_tensor, split_obs_action, soft_update
from common.buffer import ReplayBuffer
from common.logger import Logger


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax activation function"""

    def forward(self, x):
        return nn.functional.gumbel_softmax(x)


class ActorCritic(nn.Module):


    def __init__(self,
                agent_id: str,
                obs_space_dict: Dict[str, Space],
                action_space_dict: Dict[str, Space],
                hidden_sizes: List[int]=[64, 64],
                activation=nn.ReLU,
                device: str='cpu') -> None:
        super().__init__()
        self.obs_dim = dim(obs_space_dict[agent_id])
        obs_dims = list(map(lambda aid: dim(obs_space_dict[aid]), obs_space_dict))
        self.action_dim = dim(action_space_dict[agent_id])
        action_dims = list(map(lambda aid: dim(action_space_dict[aid]), action_space_dict))

        action_space = action_space_dict[agent_id]
        # continuous action
        if isinstance(action_space, Box):
            self.action_limit = action_space.high[0]
            self.mu = DeterministicPolicy(self.obs_dim, self.action_dim, hidden_sizes, 
                            activation, nn.Sigmoid, self.action_limit).to(device)
            self.mu_target = DeterministicPolicy(self.obs_dim, self.action_dim, hidden_sizes, 
                            activation, nn.Sigmoid, self.action_limit).to(device)
            for p in self.mu_target.parameters():
                p.requires_grad = False
        # discrete action
        elif isinstance(action_space, Discrete):
            self.mu = MADDPGPolicy(self.obs_dim, self.action_dim, hidden_sizes, 
                        activation, GumbelSoftmax).to(device)
            self.mu_target = deepcopy(self.mu)
        self.q = MADDPGStateActionValueFunction(obs_dims, action_dims, 
                    hidden_sizes, activation).to(device)
        self.device = device


    def step(self, observation: torch.Tensor, noise_sigma: float) -> np.ndarray:
        observation = to_tensor(observation, self.device)
        with torch.no_grad():
            try:
                epsilon = noise_sigma * np.random.randn(self.action_dim)
                action = self.mu(observation).cpu().numpy() + epsilon
                action = np.clip(action, 0, self.action_limit, dtype=np.float32)
            except AttributeError:
                epsilon = -np.log(-np.log(np.random.rand(1)))
                pass
                print('Discrete action space')
        return action


class MADDPG:
    """Multi-agent DDPG"""

    def __init__(self, args) -> None:
        set_seed(args.seed)
        self.env = make_mpe_env(args.env, continuous_actions=True)
        self.env.reset()
        self.agents = []
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for agent_id in self.env.agents:
            agent = dict()
            agent['ac'] = ActorCritic(agent_id, self.env.observation_spaces, 
                                            self.env.action_spaces, args.hidden_size, device=self.device)
            agent['mu_opt'] = Adam(agent['ac'].mu.parameters(), lr=args.lr)
            agent['q_opt'] = Adam(agent['ac'].q.parameters(), lr=args.lr)
            self.agents.append(agent)

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
        self.logger.set_saver(list(map(lambda agent: agent['ac'], self.agents)))


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
            # (B x joint_O), (B x joint_A), (B x N), (B x joint_O), (B x N)
            observations, actions, rewards, next_observations, terminations \
                = map(lambda x: x.to(self.device), self.buffer.get(self.batch_size))

            def compute_targets(observations, rewards, next_observations, terminations):
                """Compute TD targets y for Q functions"""
                # O_1, ..., O_N
                obs_dims = list(map(lambda agent: agent['ac'].obs_dim, self.agents))
                observations_ = split_obs_action(observations, obs_dims) # (B x O_1), ..., (B x O_N)

                '''
                Compute a_1', ..., a_N':
                    a_k' = (mu_target)_k(o_k)
                (B x A_1), ..., (B x A_N)
                '''
                next_actions = list(map(lambda agent, obs: agent['ac'].mu_target(obs), self.agents, observations_))
                next_actions = torch.cat(next_actions, dim=1) # (B x joint_A)

                # Q_i(o_1', ..., o_N', a_1', ..., a_N')
                Q_target_values = self.agents[i]['ac'].q(next_observations, next_actions) # (B x 1)

                '''
                Compute targets:
                    y = r_i + gamma * (1 - d_i) * Q_i(o_1', ..., o_N', a_1', ..., a_N')
                '''
                y = rewards[:, i].view(self.batch_size, 1) + self.gamma * \
                        (1 - terminations[:, i].view(self.batch_size, 1)) * Q_target_values # (B x 1)
                return y

            def compute_critic_loss(observations, actions, rewards, next_observations, terminations):
                """Compute critic loss"""
                y = compute_targets(observations, rewards, next_observations, terminations) # (B x 1)
                q_values = self.agents[i]['ac'].q(observations, actions) # (B x 1)
                q_loss = ((y - q_values) ** 2).mean() # (1,)
                return q_loss, q_values

            def compute_actor_loss(observations, actions):
                """Compute actor loss"""
                # O_1, ..., O_N
                obs_dims = list(map(lambda agent: agent['ac'].obs_dim, self.agents))
                observation = split_obs_action(observations, obs_dims)[i] # (B x O_i)

                # A_1, ..., A_N
                action_dims = list(map(lambda agent: agent['ac'].action_dim, self.agents))
                actions_ = list(split_obs_action(actions, action_dims)) # (B x A_1), ..., (B x A_N)
                actions_[i] = self.agents[i]['ac'].mu(observation).float() # (B x A_i)
                actions_ = torch.cat(actions_, dim=1) # (B x joint_A)

                mu_loss = -self.agents[i]['ac'].q(observations, actions_).mean() # (1,)
                return mu_loss
                
            self.agents[i]['q_opt'].zero_grad()
            q_loss, q_values = compute_critic_loss(observations, actions, rewards, next_observations, terminations)
            q_loss.backward()
            self.agents[i]['q_opt'].step()

            self.agents[i]['mu_opt'].zero_grad()
            mu_loss = compute_actor_loss(observations, actions)
            mu_loss.backward()
            self.agents[i]['mu_opt'].step()

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
        for i in range(self.env.num_agents):
            soft_update(self.agents[i]['ac'].mu, self.agents[i]['ac'].mu_target, self.tau)


    def act(self, i: int) -> np.ndarray:
        """
        :param i: agent index
        """
        return self.agents[i]['ac'].step(self.env.observe(agent), 0)


    def test(self) -> None:
        env = deepcopy(self.env)
        for _ in range(self.test_episodes):
            env.reset()
            total_rewards = []

            while True:
                for i, agent in enumerate(env.agents):
                    observation = env.observe(agent)
                    action = self.agents[i]['ac'].step(observation, 0)
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
                    for i, agent in enumerate(self.env.agents):
                        observations = self.env.state()
                        action = self.agents[i]['ac'].step(self.env.observe(agent), self.sigma)
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
            self.logger.render(self.act)
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
