from typing import List
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from gym.spaces import Box, Discrete


def mlp(sizes: List[int],
        activation,
        output_activation=nn.Identity):
    '''
    :param sizes: list of layers' size
    :param activation: activation layer type
    :param output_activation: output layer type
    '''
    layers = []
    for i in range(len(sizes) - 1):
        activation_ = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_()]
    return nn.Sequential(*layers)


class StochasticActor(nn.Module, ABC):


    @abstractmethod
    def _distribution(self, observation):
        '''
        Get action probability distribution
        :param observation: observation
        :return: action probability distribution
        '''
        pass


    @abstractmethod
    def _log_prob(self, pi, action):
        '''
        Compute log probility of the action
        :param pi: action probability
        :param action: action
        :return: log of action probability
        '''
        pass


    def forward(self, observation, action=None):
        pi = self._distribution(observation)
        log_prob = self._log_prob(pi, action) if action is not None else None
        return pi, log_prob


    def step(self, observation):
        with torch.no_grad():
            pi = self._distribution(observation)
            action = pi.sample()
            log_prob = self._log_prob(pi, action)
        return action, log_prob


class MLPCategoricalActor(StochasticActor):


    def __init__(self, 
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation):
        '''
        :param obs_dim: observation dimensionality
        :param action_dim: action dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.logits_network = mlp([obs_dim, *hidden_sizes, action_dim], activation)


    def _distribution(self, observation):
        return Categorical(logits=self.logits_network(observation))


    def _log_prob(self, pi, action):
        return pi.log_prob(action)


class MLPDiagGaussianActor(StochasticActor):


    def __init__(self, 
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation):
        '''
        :param obs_dim: observation dimensionality
        :param action_dim: action dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.mu = mlp([obs_dim, *hidden_sizes, action_dim], activation)
        self.log_sigma = nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dim), dtype=torch.float32))


    def _distribution(self, observation):
        mu = self.mu(observation)
        sigma = torch.exp(self.log_sigma)
        return Normal(mu, sigma)


    def _log_prob(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)


class MLPDeterministicActor(nn.Module):


    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation,
                action_limit):
        super().__init__()
        self.pi_network = mlp([obs_dim, *hidden_sizes, action_dim], activation, nn.Tanh)
        self.action_limit = action_limit


    def forward(self, observation):
        return self.action_limit * self.pi_network(observation)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPStochasticDiagGaussianActor(nn.Module):


    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, action_limit):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_sigma_layer = nn.Linear(hidden_sizes[-1], action_dim)
        # self.mu = mlp([obs_dim, *hidden_sizes, action_dim], activation, activation)
        # self.log_sigma = mlp([obs_dim, *hidden_sizes, action_dim], activation, activation)
        self.action_limit = action_limit


    def forward(self, observation, deterministic=False, need_logprob=True):
        # mu = self.mu(observation)
        # log_sigma = self.log_sigma(observation)
        output = self.net(observation)
        mu = self.mu_layer(output)
        log_sigma = self.log_sigma_layer(output)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        pi_u = Normal(mu, sigma)

        # Get unbounded action
        if deterministic:
            u = mu
        else:
            u = pi_u.rsample()

        # Bound action into [-1, 1] by applying tanh function
        if need_logprob:
            '''
            Compute log(p_a) from log(p_u) as:
            As a = tanh(u)
            => da/du = diag(1 - tanh(u_i)^2)    # Diagonal matrix w/ M_ii = 1-tanh(u)^2
            => det(da/du) = prod_{i}(1 - tanh(u_i)^2)

            p_a = p_u / det(da/du)
            => log(p_a) = log(p_u) - sum_{i}(log(1 - tanh(u_i)^2))              (1)

            Using tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            => 1 - tanh(u_i)^2 = 4 / (exp(u_i) + exp(-u_i))^2
                               = 4 / (exp(u_i) * (1 + exp(-2u_i)))^2
            => log(1 - tanh(u_i)^2) = 2 * (log2 - u - log(1 + exp(-2u_i)))
                                    = 2 * (log2 - u - softplus(-2u_i))      (2)
            (1),(2) => log(p_a) = log(p_u) - sum_{i}(2 * (log2 - u - softplus(-2u_i)))
            '''
            logp_u = pi_u.log_prob(u).sum(axis=-1)
            logp_a = logp_u - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1)
            logp_a -= np.log(self.action_limit)
        else:
            logp_a = None
        a = self.action_limit * torch.tanh(u)
        return a, logp_a


class MLPVFunction(nn.Module):


    def __init__(self,
                obs_dim,
                hidden_sizes,
                activation):
        '''
        Critic for optimizing state value function
        :param obs_dim: observation dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.value_network = mlp([obs_dim, *hidden_sizes, 1], activation)


    def forward(self, observation):
        '''
        :param observation: observation
        '''
        return torch.squeeze(self.value_network(observation), -1)


    def step(self, observation):
        with torch.no_grad():
            value = self.value_network(observation)
        return value


class MLPQFunction(nn.Module):


    def __init__(self,
                obs_dim,
                action_dim,
                hidden_sizes,
                activation):
        '''
        Critic for optimizing state-action value function
        :param obs_dim: observation dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.value_network = mlp([obs_dim + action_dim, *hidden_sizes, 1], activation)


    def forward(self, observation, action):
        '''
        :param observation: observation
        :param action: action
        '''
        obs_action = torch.cat([observation, action], dim=-1)
        return torch.squeeze(self.value_network(obs_action), -1)


class MLPStochasticActorCritic(nn.Module):


    def __init__(self, 
                observation_space,
                action_space,
                hidden_sizes=[64, 64],
                activation=nn.Tanh):
        '''
        :param observation_space: observation space
        :param action_space: action space
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.actor = MLPDiagGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.actor = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.critic = MLPVFunction(obs_dim, hidden_sizes, activation)


    def step(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        action, log_prob = self.actor.step(observation)
        value = self.critic.step(observation)
        return action.numpy(), log_prob.numpy(), value.numpy()


    def act(self, observation):
        return self.step(observation)[0]


class MLPDeterministicActorCritic(nn.Module):


    def __init__(self,
                observation_space,
                action_space,
                hidden_sizes=[256, 256],
                activation=nn.ReLU):
        '''
        :param observation_space: observation space
        :param action_space: action space
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.action_limit = action_space.high[0]

        self.actor = MLPDeterministicActor(obs_dim, self.action_dim, hidden_sizes, activation, self.action_limit)
        self.critic = MLPQFunction(obs_dim, self.action_dim, hidden_sizes, activation)


    def step(self, observation, noise_sigma):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            epsilon = noise_sigma * np.random.randn(self.action_dim)
            action = self.actor(observation) + epsilon
        return np.clip(action, -self.action_limit, self.action_limit)


    def act(self, observation):
        return self.step(observation, 0)


class MLPSACActorCritic(nn.Module):


    def __init__(self,
                observation_space,
                action_space,
                hidden_sizes=[256, 256],
                activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        action_limit = action_space.high[0]
        
        self.pi = MLPStochasticDiagGaussianActor(obs_dim, action_dim, hidden_sizes, activation, action_limit)
        self.q1 = MLPQFunction(obs_dim, action_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, action_dim, hidden_sizes, activation)


    def act(self, observation, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action, _ = self.pi(observation, deterministic, False)
        return action.numpy()
