from collections import deque
import numpy as np


class Buffer:


    def __init__(self, size: int, gamma: float, lambda_: float):
        '''
        :param size: horizon time
        :param gamma: discount factor
        :param lambda_: eligibility trace
        '''
        self._gamma = gamma
        self._lambda = lambda_
        self._trajectory = deque(maxlen=size)


    def add(self, 
            observation,
            action: int,
            reward: float,
            terminated: bool,
            value: float):
        '''
        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param terminated: whether :param observation: is the termination
        :param value: value of :param observation:
        '''
        exp = (observation, action, reward, value)
        self._trajectory.append(exp)
        if terminated or len(self._trajectory) == self._trajectory.maxlen:
            self._trajectory_len = len(self._trajectory)


    def get(self):
        '''
        :return observations: obserations along trajectory
        :return actions: actions along trajectory
        :return advs: advantage functions along trajectory
        :return rewards_to_go: rewards-to-go along trajectory
        '''
        assert self._trajectory_len, 'Need more rollouts!'

        next_value = 0
        gae = 0
        reward_to_go = 0
        advs = []
        rewards_to_go = []
        observations = []
        actions = []
        while self._trajectory:
            observation, action, reward, value = self._trajectory.pop()
            delta = reward + self._gamma * next_value - value
            next_value = value

            observations.insert(0, observation)
            actions.insert(0, action)

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
        trajectory_data = (observations, actions, advs, rewards_to_go)

        return trajectory_data
