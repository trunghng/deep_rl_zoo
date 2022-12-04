import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class DeepQNet(nn.Module):


    def __init__(self, 
                state_dim: int,
                n_actions: int,
                seed: int):
        '''
        Parameters
        ----------
        state_dim: state dimensionality
        n_actions: number of actions
        '''
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )


    def forward(self, state: np.ndarray) -> np.ndarray:
        '''
        Returns
        -------
        action values corresponding to @state
        '''
        return self.layers(state)


class CNNDeepQNet(nn.Module):


    def __init__(self,
                input_dim,
                n_actions: int,
                seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_dim), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _feature_size(self, input_dim):
        return self.features(Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)


    def forward(self, observation):
        outputs = self.features(observation)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)

        return outputs


class DuelingQNet(nn.Module):


    def __init__(self,
                input_dim,
                n_actions: int,
                seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_dim), 512),
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
            nn.Linear(256, n_actions)
        )

    def _feature_size(self, input_dim):
        return self.features(Variable(torch.zeros(1, *input_dim))).view(1, -1).size(1)


    def forward(self, observation):
        outputs = self.features(observation)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)

        state_value = self.state_value(outputs)
        advantage = self.advantage(outputs)

        return state_value + advantage - advantage.mean(dim=1, keepdims=True)
