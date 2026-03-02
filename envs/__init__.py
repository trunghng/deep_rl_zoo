from gymnasium.envs.registration import register

register(
    id='Bipedal-v1',
    entry_point='envs.biped.biped_env:Bipedal',
    max_episode_steps=1000,
)