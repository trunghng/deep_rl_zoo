import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DeepQNetwork(nn.Module):


    def __init__(self, 
                state_size: int,
                action_size: int,
                layers_size: List[int],
                seed: int):
        '''
        Parameters
        ----------
        state_size: state size
        action_size: action size, = #actions
        layers_size: sizes of hidden layers
        seed: random seed
        '''
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for i in range(len(layers_size)):
            if i == 0:
                layers.append(nn.Linear(state_size, layers_size[i]))
            else:
                layers.append(nn.Linear(layers_size[i - 1], layers_size[i]))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(layers_size[-1], action_size)


    def forward(self, state: np.ndarray) -> np.ndarray:
        '''
        Returns
        -------
        action values corresponding to @state
        '''
        outputs = F.relu(self.layers[0](state))
        for fc in self.layers[1:]:
            outputs = F.relu(fc(outputs))
        action_values = self.output_layer(outputs)

        return action_values
