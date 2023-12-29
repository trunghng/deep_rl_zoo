from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal

from common.network import *


class Policy(nn.Module, ABC):


    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor=None):
        """Policy network forward path"""


class DeterministicPolicy(Policy):

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module,
                output_activation: nn.Module,
                action_limit: float) -> None:
        super().__init__()
        self.network = MLP([obs_dim, *hidden_sizes, action_dim], activation, output_activation)
        self.action_limit = action_limit


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.network(observation)


class StochasticPolicy(Policy):


    @abstractmethod
    def _distribution(self, observation: torch.Tensor) -> Distribution:
        """Returns the probability distribution over actions

        :param observation: current observation
        """


    @abstractmethod
    def _log_prob(self,
                pi: Distribution,
                action: torch.Tensor) -> torch.Tensor:
        """Returns the log-likelihood of :param action:

        :param pi: action probability distribution
        :param action: action taken
        """


    def forward(self,
                observation: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self._distribution(observation)
        log_prob = self._log_prob(pi, action)
        return pi, log_prob


class CategoricalPolicy(StochasticPolicy):
    """Categorical policy, used for discrete action space

    :param obs_dim: observation dimensionality
    :param action_dim: action dimensionality
    :param hidden_sizes: list of hidden layers' size
    :param activation: activation function
    """
    
    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module) -> None:
        super().__init__()
        self.logits = MLP([obs_dim, *hidden_sizes, action_dim], activation)


    def _distribution(self, observation: torch.Tensor) -> Distribution:
        return Categorical(logits=self.logits(observation))


    def _log_prob(self,
                pi: Distribution,
                action: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(action)


class DiagonalGaussianPolicy(StochasticPolicy):
    """Diagonal Gaussian policy, used for continuous action space

    :param obs_dim: observation dimensionality
    :param action_dim: action dimensionality
    :param hidden_sizes: list of hidden layers' size
    :param activation: activation function
    """

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module) -> None:
        super().__init__()
        self.mu = MLP([obs_dim, *hidden_sizes, action_dim], activation)
        self.log_sigma = nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dim), dtype=torch.float32))


    def _distribution(self, observation: torch.Tensor) -> Distribution:
        mu = self.mu(observation)
        sigma = torch.exp(self.log_sigma)
        return Normal(mu, sigma)


    def _log_prob(self,
                pi: Distribution,
                action: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(action).sum(axis=-1)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SoftPolicy(Policy):
    """Diagonal Gaussian policy, used for Soft Actor-Critic

    :param obs_dim: observation dimensionality
    :param action_dim: action dimensionality
    :param hidden_sizes: list of hidden layers' size
    :param activation: activation function
    :action_limit: action limit
    """

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module,
                action_limit: float) -> None:
        super().__init__()
        self.network = MLP([obs_dim, *hidden_sizes], activation, activation)
        self.mu_outlayer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_sigma_outlayer = nn.Linear(hidden_sizes[-1], action_dim)
        self.action_limit = action_limit


    def forward(self, observation, deterministic=False, need_logprob=True) -> Tuple[torch.Tensor, np.ndarray]:
        output = self.network(observation)
        mu = self.mu_outlayer(output)
        log_sigma = self.log_sigma_outlayer(output)
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
            => da/du = diag(1 - tanh(u_i)^2)    # Diagonal matrix w/ M_ii = 1 - tanh(u)^2
            => det(da/du) = prod_{i}(1 - tanh(u_i)^2)

            p_a = p_u / det(da/du)
            => log(p_a) = log(p_u) - sum_{i}(log(1 - tanh(u_i)^2))          (1)

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


class DiscretePolicy(Policy):
    """Policy used for discrete action space"""

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module,
                output_activation: nn.Module) -> None:
        super().__init__()
        self.logits = MLP([obs_dim, *hidden_sizes, action_dim], activation, output_activation)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.logits(observation)
