import random
from typing import Callable, List

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
import pettingzoo.mpe as mpe
import torch
import numpy as np


def to_tensor(ndarray, device):
    return torch.as_tensor(ndarray, device=device, dtype=torch.float32)


def set_seed(seed: int) -> None:
    """Set seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def flatten(tensor):
    """Flatten tensor"""

    return torch.cat([t.contiguous().view(-1) for t in tensor])


def conjugate_gradient(Ax, b, cg_iters: int):
    """Conjugate gradient"""

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


def soft_update(network, target_network, tau: float) -> None:
    """
    Perform soft (Polyak average) update on target network's parameters
        p_target = (1 - tau) * p_target + tau * p

    :param params: parameters used for updating `target_params`
    :param target_params: parameters to update
    :param tau: soft update coefficient (in [0, 1])
    """
    with torch.no_grad():
        for p, p_target in zip(network.parameters(), target_network.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)


def hard_update(network, target_network) -> None:
    """Perform hard update on target network's parameters"""
    target_network.load_state_dict(network.state_dict())


def dim(space: Space) -> int:
    """Return dimensionality of the space"""
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    else:
        pass


def make_atari_env(env: str, render_mode: str=None) -> gym.Env:
    """Create atari environment

    :param env: environment ID
    :param render_mode: render mode, specify when enabling render
    """
    if render_mode:
        env = gym.make(env, render_mode=render_mode)
    else:
        gym.make(env)
    env = AtariPreprocessing(env, grayscale_newaxis=True)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)))
    assert len(env.observation_space.shape) == 3, f'{env.observation_space.shape} is not an image space'
    h, w, c = env.observation_space.shape
    env.observation_space = Box(low=0, high=255, shape=(c, h, w), dtype=env.observation_space.dtype)

    return env


def make_mpe_env(env: str, **kwargs) -> Callable:
    """Create PettingZoo MPE environment

    :param env: environment ID
    """
    env_dict = {
        'adversary': mpe.simple_adversary_v3,
        'crypto': mpe.simple_crypto_v3,
        'push': mpe.simple_push_v3,
        'reference': mpe.simple_reference_v3,
        'speaker-listener': mpe.simple_speaker_listener_v4,
        'spread': mpe.simple_spread_v3,
        'tag': mpe.simple_tag_v3,
        'world-comm': mpe.simple_world_comm_v3,
        'simple': mpe.simple_v3
    }

    return env_dict[env].env(**kwargs)


def split_obs_action(joint_data: torch.Tensor, data_dims: List[int]):
    """Split action (orbservation) from joint action (observation),
            used for multi-agent method

    :param joint_data: joint action (observation) of agents
    :param data_dims: list of dimensionalities of agents' action (observation) spaces
    """
    return torch.split(joint_data, data_dims, dim=1)
