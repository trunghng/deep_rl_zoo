from typing import List
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
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


class MLPGaussianActor(StochasticActor):


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
        self.mean = mlp([obs_dim, *hidden_sizes, action_dim], activation)
        self.log_std = nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dim), dtype=torch.float32))


    def _distribution(self, observation):
        mean = self.mean(observation)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


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


class MLPVCritic(nn.Module):


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


class MLPQCritic(nn.Module):


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
            self.actor = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.actor = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.critic = MLPVCritic(hidden_sizes, activation, obs_dim)


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
        self.action_high = action_space.high[0]
        self.action_low = action_space.low[0]

        self.actor = MLPDeterministicActor(obs_dim, self.action_dim, hidden_sizes, activation, self.action_high)
        self.critic = MLPQCritic(obs_dim, self.action_dim, hidden_sizes, activation)


    def act(self, observation, noise_sigma):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            epsilon = noise_sigma * np.random.randn(self.action_dim)
            action = self.actor(observation) + epsilon
        return np.clip(action, self.action_low, self.action_high)
