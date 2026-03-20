from gymnasium.envs.registration import register

from envs.biped.biped_config import BipedConfig
from envs.biped.biped_walking_config import BipedWalkingConfig, BipedTeacherConfig, BipedVisionConfig

register(
    id='Bipedal-v1',
    entry_point='envs.biped.biped_env:Bipedal',
    kwargs={'config': BipedConfig()},
    max_episode_steps=1000,
)

register(
    id='BipedalWalking-v1',
    entry_point='envs.biped.biped_env:Bipedal',
    kwargs={'config': BipedWalkingConfig()},
    max_episode_steps=1000,
)

register(
    id='BipedalTeacher-v1',
    entry_point='envs.biped.biped_env:Bipedal',
    kwargs={'config': BipedTeacherConfig()},
    max_episode_steps=1000,
)

register(
    id='BipedalVision-v1',
    entry_point='envs.biped.biped_env:Bipedal',
    kwargs={'config': BipedVisionConfig()},
    max_episode_steps=1000,
)
