import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):


    def __init__(self, 
                state_size: int,
                action_size: int,
                lr: float,
                seed: int):
        '''
        Parameters
        ----------
        state_size: state size
        action_size: action size, = #actions
        lr: learning rate
        seed: random seed
        '''
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)


    def forward(self, state: np.ndarray) -> np.ndarray:
        '''
        Returns
        -------
        action values corresponding to @state
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_values = self.fc3(x)

        return action_values
