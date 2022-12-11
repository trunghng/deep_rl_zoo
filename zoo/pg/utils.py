from collections import deque
import numpy as np
import torch


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
    r = b - Ax(x)
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
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


    def add(self, 
            observation,
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            terminated: bool):
        '''
        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param value: value of :param observation:
        :param log_prob:
        :param terminated: whether :param observation: is the termination
        '''
        exp = (observation, action, reward, value, log_prob)
        self._trajectory.append(exp)
        if terminated or len(self._trajectory) == self._trajectory.maxlen:
            self._trajectory_len = len(self._trajectory)


    def get(self):
        '''
        :return observations: obserations along trajectory
        :return actions: actions along trajectory
        :return log_probs:
        :return advs: advantage functions along trajectory
        :return rewards_to_go: rewards-to-go along trajectory
        '''
        assert self._trajectory_len, 'Need more rollouts!'

        next_value = 0
        gae = 0
        reward_to_go = 0
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
                           torch.as_tensor(actions, dtype=torch.int32),
                           torch.as_tensor(log_probs, dtype=torch.float32),
                           torch.as_tensor(advs, dtype=torch.float32),
                           torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return trajectory_data
