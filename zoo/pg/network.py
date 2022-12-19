from typing import List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np
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


class Actor(nn.Module, ABC):


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


class MLPCategoricalActor(Actor):


    def __init__(self, 
                obs_dim: int,
                act_dim: int,
                hidden_sizes: List[int],
                activation):
        '''
        :param obs_dim: observation dimensionality
        :param act_dim: action dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.logits_network = mlp([obs_dim, *hidden_sizes, act_dim], activation)


    def _distribution(self, observation):
        return Categorical(logits=self.logits_network(observation))


    def _log_prob(self, pi, action):
        return pi.log_prob(action)


class MLPGaussianActor(Actor):


    def __init__(self, 
                obs_dim: int,
                act_dim: int,
                hidden_sizes: List[int],
                activation):
        '''
        :param obs_dim: observation dimensionality
        :param act_dim: action dimensionality
        :param hidden_sizes: list of hidden layers' size
        :param activation: activation function
        '''
        super().__init__()
        self.mean = mlp([obs_dim, *hidden_sizes, act_dim], activation)
        self.log_std = nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim), dtype=torch.float32))


    def _distribution(self, observation):
        mean = self.mean(observation)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


    def _log_prob(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)


class MLPCritic(nn.Module):


    def __init__(self,
                obs_dim,
                hidden_sizes,
                activation):
        '''
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


class MLPActorCritic(nn.Module):


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

        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)


    def step(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        action, log_prob = self.actor.step(observation)
        value = self.critic.step(observation)
        return action.numpy(), log_prob.numpy(), value.numpy()


    def act(self, observation):
        return self.step(observation)[0]
