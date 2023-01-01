from collections import deque
import random
import numpy as np
import torch
from common.mpi_utils import mpi_get_statistics


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


def polyak_update(params, target_params, tau):
    '''
    Perform Polyak average update on `target_params` using `params`:
        p_target = (1 - tau) * p_target + tau * p

    :param params: parameters used for updating `target_params`
    :param target_params: parameters to update
    :param tau: (float) soft update coefficient (in [0, 1])
    '''
    with torch.no_grad():
        for p, p_target in zip(params, target_params):
            p_target.data.copy_((1 - tau) * p_target.data + tau * p.data)


class Buffer:


    def __init__(self, size: int, gamma: float, lamb: float):
        '''
        :param size: Max size
        :param gamma: Discount factor
        :param lamb: Lambda for GAE-Lambda
        '''
        self.gamma = gamma
        self.lamb = lamb
        self.trajectories = deque(maxlen=size)
        self.trajectories_start_idx = [0]
        self.last_values = []


    def finish_rollout(self, last_value):
        self.last_values.append(last_value)
        if len(self.trajectories) < self.trajectories.maxlen:
            self.trajectories_start_idx.append(len(self.trajectories))


    def add(self, observation, action, reward, value, log_prob):
        '''
        :param observation: observation
        :param action: action taken at :param observation:
        :param reward: corresponding reward for taking :param action:
        :param value: value of :param observation:
        :param log_prob: log probability
        '''
        if action.size == 1:
            action = float(action)
        exp = (observation, action, reward, value, log_prob)
        self.trajectories.append(exp)


    def get(self):
        '''
        :return observations: obserations across trajectories
        :return actions: actions across trajectories
        :return log_probs: log probabilities across trajectories
        :return advs: advantage functions across trajectories
        :return rewards_to_go: rewards-to-go across trajectories
        '''
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
        mean, std = mpi_get_statistics(advs)
        advs = (advs - mean) / std
        rewards_to_go = np.array(rewards_to_go)

        rollout_data = (torch.as_tensor(observations, dtype=torch.float32),
                       torch.as_tensor(actions, dtype=torch.float32),
                       torch.as_tensor(log_probs, dtype=torch.float32),
                       torch.as_tensor(advs, dtype=torch.float32),
                       torch.as_tensor(rewards_to_go, dtype=torch.float32))
        return rollout_data


class ReplayBuffer:


    def __init__(self, size, obs_dim, action_dim):
        '''
        Experience replay buffer

        :param size: (int) Buffer size
        :param obs_dim:
        :param action_dim:
        '''
        self.obs_buffer = np.zeros(self._to_shape(size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(self._to_shape(size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.nextobs_buffer = np.zeros(self._to_shape(size, obs_dim), dtype=np.float32)
        self.terminated_buffer = np.zeros(size, dtype=np.float32)
        self.current_idx = self.size = 0
        self.max_size = size


    def _to_shape(self, length, dim):
        return (length, dim) if np.isscalar(dim) else (length, *dim)


    def add(self, observation, action, reward, next_observation, terminated):
        '''
        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param terminated:
        '''
        self.obs_buffer[self.current_idx] = observation
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.nextobs_buffer[self.current_idx] = next_observation
        self.terminated_buffer[self.current_idx] = terminated
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        '''
        Sample a minibatch of experiences with size :param batch_size:

        :param batch_size: (int) Minibatch size
        '''
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch_data = (torch.as_tensor(self.obs_buffer[idxs], dtype=torch.float32),
                      torch.as_tensor(self.action_buffer[idxs], dtype=torch.float32),
                      torch.as_tensor(self.reward_buffer[idxs], dtype=torch.float32),
                      torch.as_tensor(self.nextobs_buffer[idxs], dtype=torch.float32),
                      torch.as_tensor(self.terminated_buffer[idxs], dtype=torch.float32))
        return batch_data
