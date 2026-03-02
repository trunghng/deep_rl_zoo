from abc import ABC, abstractmethod
from collections import deque
import random
from typing import Tuple

from gymnasium.core import ActType, ObsType
import numpy as np
import torch

from common.mpi_utils import mpi_get_statistics


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
    :param lamb: Lambda for GAE-Lambda
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

        TD error: delta_t = r_t + gamma * V(s_t+1) - V(s_t)
        GAE = sum_{i=0}^{infty} (gamma * lambda)^i * delta_t+i

        :return observations: observations across trajectories
        :return actions: actions across trajectories
        :return log_probs: log probabilities across trajectories
        :return advs: advantage functions across trajectories
        :return rewards_to_go: rewards-to-go across trajectories
        """
        assert len(self.trajectories) == self.trajectories.maxlen, 'Need more rollouts!'

        observations, actions, log_probs, advs, rewards_to_go = [], [], [], [], []
        while self.trajectories_start_idx:
            next_value = self.last_values.pop(-1)
            trajectory_start_idx = self.trajectories_start_idx.pop(-1)
            gae = 0
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
                rewards_to_go_.insert(0, float(gae + value))
            observations += observations_
            actions += actions_
            log_probs += log_probs_
            advs += advs_
            rewards_to_go += rewards_to_go_

        self.trajectories_start_idx = [0]
        observations = np.array(observations)
        actions = np.array(actions)
        advs = np.array(advs)
        mean, std = mpi_get_statistics(advs)
        advs = (advs - mean) / std
        rewards_to_go = np.array(rewards_to_go)

        rollout_data = (torch.as_tensor(observations, dtype=torch.float32),
                       torch.as_tensor(actions, dtype=torch.float32),
                       torch.as_tensor(log_probs, dtype=torch.float32),
                       torch.as_tensor(advs, dtype=torch.float32),
                       torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return rollout_data


class VectorRolloutBuffer(Buffer):
    """Buffer used in on-policy methods

    :param obs_dim: observation's dimension
    :param action_dim: action's dimension
    :param size: size allocated for each buffer
    :param n_envs: Number of parallel envs
    :param gamma: Discount factor
    :param lamb: Lambda for GAE-Lambda
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        size: int,
        n_envs: int,
        gamma: float,
        lamb: float
    ) -> None:
        self.observations = np.zeros((size, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, n_envs), dtype=np.float32)
        self.values = np.zeros((size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((size, n_envs), dtype=np.float32)
        self.advs = np.zeros((size, n_envs), dtype=np.float32)
        self.rewards_to_go = np.zeros((size, n_envs), dtype=np.float32)
        self.masks = np.zeros((size, n_envs), dtype=np.float32)
        self.gamma = gamma
        self.lamb = lamb
        self.pointer = self.start_idx = 0
        self.max_size = size

    def finish_rollout(self, last_values):
        """
        Call once a rollout finishes

        TD error: delta_t = r_t + gamma * V(s_t+1) - V(s_t)
        GAE = sum_{i=0}^{infty} (gamma * lambda)^i * delta_t+i
        """
        next_values = last_values
        next_gaes = np.zeros_like(last_values)
        
        for t in reversed(range(self.start_idx, self.pointer)):
            delta = self.rewards[t] + self.gamma * next_values * self.masks[t] - self.values[t]
            next_values = self.values[t]
            next_gaes = (self.gamma * self.lamb) * next_gaes + delta

            self.advs[t] = next_gaes
            self.rewards_to_go[t] = self.advs[t] + self.values[t]
        self.start_idx = self.pointer

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray
    ) -> None:
        if self.pointer == self.max_size:
            self.pointer = 0
        self.observations[self.pointer] = observations
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = rewards
        self.values[self.pointer] = values
        self.log_probs[self.pointer] = log_probs
        self.masks[self.pointer] = 1.0 - terminated
        self.pointer += 1

    def get(self):
        """
        Get data from the buffer

        :return observations: observations across trajectories
        :return actions: actions across trajectories
        :return log_probs: log probabilities across trajectories
        :return advs: advantage functions across trajectories
        :return rewards_to_go: rewards-to-go across trajectories
        """
        assert self.pointer == self.max_size, 'Need more rollouts!'

        mean, std = mpi_get_statistics(self.advs)
        self.advs = (self.advs - mean) / std

        observations = self.observations.reshape(-1, self.observations.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        log_probs = self.log_probs.reshape(-1)
        advs = self.advs.reshape(-1)
        rewards_to_go = self.rewards_to_go.reshape(-1)
        self.pointer = self.start_idx = 0

        return (
            torch.as_tensor(observations, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(log_probs, dtype=torch.float32),
            torch.as_tensor(advs, dtype=torch.float32),
            torch.as_tensor(rewards_to_go, dtype=torch.float32)
        )


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
