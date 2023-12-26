from typing import List

import torch
import torch.nn as nn

from common.network import *


class ValueFunction(nn.Module):
    """Value function base class"""


class StateValueFunction(ValueFunction):
    """State value function (V function)"""

    def __init__(self,
                obs_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module) -> None:
        super().__init__()
        self.value_network = MLP([obs_dim, *hidden_sizes, 1], activation)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_network(observation)


class StateActionValueFunction(ValueFunction):
    """State-action value function (Q function)"""

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int],
                activation: nn.Module) -> None:
        super().__init__()
        self.value_network = MLP([obs_dim + action_dim, *hidden_sizes, 1], activation)


    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat([observation, action], dim=-1)
        return self.value_network(obs_action)


class MADDPGStateActionValueFunction(ValueFunction):
    """"""

    def __init__(self,
                obs_dims: List[int],
                action_dims: List[int],
                hidden_sizes: List[int],
                activation: nn.Module) -> None:
        super().__init__()
        self.value_network = MLP([sum(obs_dims) + sum(action_dims), *hidden_sizes, 1], activation)


    def forward(self,
                joint_observation: torch.Tensor,
                joint_action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((joint_observation, joint_action), dim=-1)
        return self.value_network(obs_action)
 

class MLPDeepQNet(ValueFunction):
    """MLP value network for DQN"""

    def __init__(self,
                obs_dim: int,
                action_dim: int,
                hidden_sizes: List[int]=[64, 64],
                activation=nn.ReLU) -> None:
        super().__init__()
        self.value_network = MLP([obs_dim, *hidden_sizes, action_dim], activation)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_network(observation)


class CNNDeepQNet(ValueFunction):

    def __init__(self,
                obs_dim: int,
                action_dim: int):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(obs_dim), 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )


    def _feature_size(self, obs_dim):
        return self.features(Variable(torch.zeros(1, *obs_dim))).view(1, -1).size(1)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output = self.features(observation)
        value = self.fc(outputs.view(outputs.size(0), -1))
        return value


class MLPDuelingQNet(ValueFunction):

    def __init__(self,
                obs_dim: int,
                action_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(obs_dim), 512),
            nn.ReLU()
        )
        # State-value stream
        self.state_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def _feature_size(self, obs_dim):
        return self.features(torch.autograd.Variable(torch.zeros(1, *obs_dim))).view(1, -1).size(1)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output = self.features(observation)
        output = self.fc(output.view(output.size(0), -1))

        state_value = self.state_value(output)
        advantage = self.advantage(output)

        return state_value + advantage - advantage.mean(dim=1, keepdims=True)
