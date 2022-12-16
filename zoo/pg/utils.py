from collections import deque
import numpy as np
import torch
from os.path import dirname, join, realpath
import sys
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
from common.mpi_utils import mpi_mean_std
from copy import deepcopy


def flatten(tensor):
    '''
    Flatten tensor
    '''
    return torch.cat([t.contiguous().view(-1) for t in tensor])


def conjugate_gradient(Ax, b, cg_iters: int):
    '''
    Conjugate gradient
    '''
    x = torch.zeros(b.shape)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for i in range(cg_iters):
        Ap = Ax(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rdotr_new = torch.dot(r, r)
        p = r + rdotr_new / rdotr * p
        rdotr = rdotr_new
    return x


class Buffer:


    def __init__(self, size: int, gamma: float, lambda_: float):
        '''
        :param size: Horizon time
        :param gamma: Discount factor
        :param lambda_: Lambda for GAE
        '''
        self._gamma = gamma
        self._lambda = lambda_
        self._trajectory = deque(maxlen=size)


    def finish_rollout(self, last_value):
        self._last_value = last_value
        self._trajectory_len = len(self._trajectory)


    def add(self, 
            observation,
            action: float,
            reward: float,
            value: float,
            log_prob: float):
        '''
        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param value: value of :param observation:
        :param log_prob:
        '''
        if action.size == 1:
            action = float(action)
        exp = (observation, action, reward, value, log_prob)
        self._trajectory.append(exp)


    def get(self):
        '''
        :return observations: observations along trajectory
        :return actions: actions along trajectory
        :return log_probs: log probabilities along trajectory
        :return advs: advantage functions along trajectory
        :return rewards_to_go: rewards-to-go along trajectory
        '''
        assert self._trajectory_len, 'Need more rollouts!'

        next_value = gae = reward_to_go = self._last_value
        observations = []
        actions = []
        log_probs = []
        advs = []
        rewards_to_go = []
        while self._trajectory:
            observation, action, reward, value, log_prob = self._trajectory.pop()
            delta = reward + self._gamma * next_value - value
            next_value = value

            observations.insert(0, observation)
            actions.insert(0, action)
            log_probs.insert(0, log_prob)

            step_to_go = self._trajectory_len - len(self._trajectory)
            gae += np.power((self._gamma * self._lambda), step_to_go) * delta
            advs.insert(0, gae)

            reward_to_go = reward + self._gamma * reward_to_go
            rewards_to_go.insert(0, reward_to_go)

        observations = np.array(observations)
        actions = np.array(actions)
        advs = np.array(advs)
        mean, std = np.mean(advs), np.std(advs)
        advs = (advs - mean) / std
        rewards_to_go = np.array(rewards_to_go)

        trajectory_data = (torch.as_tensor(observations, dtype=torch.float32),
                           torch.as_tensor(actions, dtype=torch.float32),
                           torch.as_tensor(log_probs, dtype=torch.float32),
                           torch.as_tensor(advs, dtype=torch.float32),
                           torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return trajectory_data


class MPIBuffer:


    def __init__(self, size: int, gamma: float, lambda_: float):
        '''
        :param size: max size
        :param gamma: Discount factor
        :param lambda_: Lambda for GAE
        '''
        self._gamma = gamma
        self._lambda = lambda_
        self._trajectories = deque(maxlen=size)
        self._trajectories_start_idx = [0]
        self._last_values = []


    def finish_rollout(self, last_value):
        self._last_values.append(last_value)
        if len(self._trajectories) < self._trajectories.maxlen:
            self._trajectories_start_idx.append(len(self._trajectories))


    def add(self, observation, action, reward, value, log_prob):
        '''
        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param value: value of :param observation:
        :param log_prob:
        '''
        if action.size == 1:
            action = float(action)
        exp = (observation, action, reward, value, log_prob)
        self._trajectories.append(exp)


    def get(self):
        '''
        :return observations: obserations across trajectories
        :return actions: actions across trajectories
        :return log_probs: log probabilities across trajectories
        :return advs: advantage functions across trajectories
        :return rewards_to_go: rewards-to-go across trajectories
        '''
        assert len(self._trajectories) == self._trajectories.maxlen, 'Need more rollouts!'

        observations, actions, log_probs, advs, rewards_to_go = [], [], [], [], []
        while self._trajectories_start_idx:
            next_value = gae = reward_to_go = self._last_values.pop(-1)
            trajectory_start_idx = self._trajectories_start_idx.pop(-1)
            final_step = len(self._trajectories)
            observations_, actions_, log_probs_, advs_, rewards_to_go_ = [], [], [], [], []

            while len(self._trajectories) != trajectory_start_idx:
                observation, action, reward, value, log_prob = self._trajectories.pop()
                delta = reward + self._gamma * next_value - value
                next_value = value

                observations_.insert(0, observation)
                actions_.insert(0, action)
                log_probs_.insert(0, log_prob)

                step_to_go = final_step - len(self._trajectories)
                gae += np.power((self._gamma * self._lambda), step_to_go) * delta
                advs_.insert(0, float(gae))

                reward_to_go = reward + self._gamma * reward_to_go
                rewards_to_go_.insert(0, float(reward_to_go))
            observations += observations_
            actions += actions_
            log_probs += log_probs_
            advs += advs_
            rewards_to_go += rewards_to_go_

        self._trajectories_start_idx = [0]
        observations = np.array(observations)
        actions = np.array(actions)
        advs = np.array(advs)
        mean, std = mpi_mean_std(advs)
        advs = (advs - mean) / std
        rewards_to_go = np.array(rewards_to_go)

        rollout_data = (torch.as_tensor(observations, dtype=torch.float32),
                       torch.as_tensor(actions, dtype=torch.float32),
                       torch.as_tensor(log_probs, dtype=torch.float32),
                       torch.as_tensor(advs, dtype=torch.float32),
                       torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return rollout_data
