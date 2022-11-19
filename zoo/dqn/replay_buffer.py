import numpy as np
from collections import deque
from typing import Tuple, List, NamedTuple
import random


class Experience(NamedTuple):
    
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    '''
    Replay buffer class
    '''

    def __init__(self, buffer_size: int):
        '''
        Parameters
        ----------
        buffer_size: buffer_size
        '''
        self.memory = deque(maxlen=buffer_size)


    def add(self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminated: bool) -> None:
        '''
        Save transition into the buffer memory

        Parameters
        ----------
        state: current state
        action: action taken
        reward: reward corresponding
        next_state: next state
        terminated: whether is terminated
        '''
        e = Experience(state, action, reward, next_state, terminated)
        self.memory.append(e)


    def sample(self, batch_size: int) -> Tuple[List[np.ndarray], List[int], List[float], List[np.ndarray], List[bool]]:
        '''
        Sample a mini batch randomly from the buffer memory

        Parameters
        ----------        
        batch_size: minibatch size
        '''
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, terminated = [], [], [], [], []

        for e in experiences:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            terminated.append(e.terminated)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminated)


    def __len__(self) -> int:
        return len(self.memory)

