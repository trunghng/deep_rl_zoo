import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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
        seed: random seed
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
        self.input_dim = input_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def feature_size(self):
        return self.features(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)

        return outputs
