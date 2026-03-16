from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

from common.utils import setup_headless_rendering

class BaseEnv(MujocoEnv, utils.EzPickle):

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        setup_headless_rendering()
        self.config = config

    def _get_obs(self):
        raise NotImplementedError('Subclasses must implement _get_obs')

    def _update_states(self, action, torques, observation):
        raise NotImplementedError('Subclasses must implement _update_states')

    def _get_total_reward(self) -> bool:
        raise NotImplementedError('Subclasses must implement _get_total_reward')

    def _get_termination(self) -> bool:
        raise NotImplementedError('Subclasses must implement _get_termination')

    def _get_info(self) -> dict:
        return {}

    def step(self, action):
        raise NotImplementedError('Subclasses must implement step')
