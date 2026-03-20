from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np

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

    def get_config(self) -> dict:
        def _parse_config(obj):
            config_dict = {}
            for key in dir(obj):
                if key.startswith('__'): continue

                val = getattr(obj, key)
                if isinstance(val, type):
                    config_dict[key] = _parse_config(val)
                elif callable(val):
                    continue
                elif isinstance(val, np.ndarray):
                    config_dict[key] = val.tolist()
                else:
                    config_dict[key] = val
            return config_dict
        return _parse_config(self.config)

    def step(self, action):
        raise NotImplementedError('Subclasses must implement step')
