from abc import ABC, abstractmethod
from collections import deque
import random
from typing import Tuple

from gymnasium.core import ActType, ObsType
import numpy as np
import torch


class Buffer(ABC):

    @abstractmethod
    def add(self, data) -> None:
        """Store data into the buffer"""

    @abstractmethod
    def get(self) -> Tuple:
        """Get data from the buffer"""


class RolloutBuffer(Buffer):
    """Buffer used in on-policy methods

    :param size: Max size
    :param gamma: Discount factor
    :param lamb: Lambda for GAE-Lambdae
    """

    def __init__(self, size: int, gamma: float, lamb: float) -> None:
        self.gamma = gamma
        self.lamb = lamb
        self.trajectories = deque(maxlen=size)
        self.trajectories_start_idx = [0]
        self.last_values = []


    def finish_rollout(self, last_value):
        self.last_values.append(last_value)
        if len(self.trajectories) < self.trajectories.maxlen:
            self.trajectories_start_idx.append(len(self.trajectories))


    def add(self,
            observation: ObsType,
            action: ActType,
            reward: float,
            value: float,
            log_prob: float) -> None:
        """Store data into the buffer

        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param value: value of :param observation:
        :param log_prob: log probability
        """
        if action.size == 1:
            action = float(action)
        exp = (observation, action, reward, value, log_prob)
        self.trajectories.append(exp)


    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data from the buffer

        :return observations: observations across trajectories
        :return actions: actions across trajectories
        :return log_probs: log probabilities across trajectories
        :return advs: advantage functions across trajectories
        :return rewards_to_go: rewards-to-go across trajectories
        """
        assert len(self.trajectories) == self.trajectories.maxlen, 'Need more rollouts!'

        observations, actions, log_probs, advs, rewards_to_go = [], [], [], [], []
        while self.trajectories_start_idx:
            next_value = gae = reward_to_go = self.last_values.pop(-1)
            trajectory_start_idx = self.trajectories_start_idx.pop(-1)
            final_step = len(self.trajectories)
            observations_, actions_, log_probs_, advs_, rewards_to_go_ = [], [], [], [], []

            while len(self.trajectories) != trajectory_start_idx:
                observation, action, reward, value, log_prob = self.trajectories.pop()
                delta = reward + self.gamma * next_value - value
                next_value = value

                observations_.insert(0, observation)
                actions_.insert(0, action)
                log_probs_.insert(0, log_prob)

                step_to_go = final_step - len(self.trajectories)
                gae = (self.gamma * self.lamb) * gae + delta
                advs_.insert(0, float(gae))

                reward_to_go = reward + self.gamma * reward_to_go
                rewards_to_go_.insert(0, float(reward_to_go))
            observations += observations_
            actions += actions_
            log_probs += log_probs_
            advs += advs_
            rewards_to_go += rewards_to_go_

        self.trajectories_start_idx = [0]
        observations = np.array(observations)
        actions = np.array(actions)
        advs = np.array(advs)
        # mean, std = mpi_get_statistics(advs)
        # advs = (advs - mean) / std
        rewards_to_go = np.array(rewards_to_go)

        rollout_data = (torch.as_tensor(observations, dtype=torch.float32),
                       torch.as_tensor(actions, dtype=torch.float32),
                       torch.as_tensor(log_probs, dtype=torch.float32),
                       torch.as_tensor(advs, dtype=torch.float32),
                       torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return rollout_data


class ReplayBuffer(Buffer):
    """Buffer used in off-policy methods

    :param size: buffer size
    """

    def __init__(self, size: int) -> None:
        self.memory = deque(maxlen=size)


    def __len__(self) -> int:
        return len(self.memory)


    def add(self,
            observation: ObsType,
            action: ActType,
            reward: float,
            next_observation: ObsType,
            terminated: bool) -> None:
        self.memory.append((observation, action, reward, next_observation, terminated))


    def get(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch of data randomly from the buffer

        :param batch_size: mini-batch size
        """
        experiences = random.sample(self.memory, k=batch_size)
        # batch_data = [observations, actions, rewards, next_observations, terminated]
        batch_data = [[], [], [], [], []]

        for experience in experiences:
            for data, data_buffer in zip(experience, batch_data):
                data_buffer.append(data)

        for i in range(len(batch_data)):
            batch_data[i] = torch.as_tensor(np.asarray(batch_data[i]), dtype=torch.float32)
            if len(batch_data[i].shape) == 1:
                batch_data[i] = batch_data[i].view(batch_size, 1)
        return tuple(batch_data)
