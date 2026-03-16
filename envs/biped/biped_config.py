import numpy as np
from envs.base.base_config import BaseLeggedConfig

class BipedConfig(BaseLeggedConfig):

    class init_state(BaseLeggedConfig.init_state):
        pos = [0.0, 0.0, 1.15]

        # The "athletic stance" (rad)
        # Order: hip_x, hip_z, hip_y, knee, ankle_x, ankle_y (left leg then right leg)
        default_joint_angles = np.array([
            0.0, 0.0, -0.2, 0.4, 0.0, -0.2, # left
            0.0, 0.0, -0.2, 0.4, 0.0, -0.2  # right
        ])

    class asset(BaseLeggedConfig.asset):
        file_name = 'bipedal.xml'

    class control(BaseLeggedConfig.control):
        stiffness = {
            'hip_x': 40.0,   # Side-to-side hips need less strength
            'hip_z': 40.0,   # Twisting hips need less strength
            'hip_y': 120.0,  # Forward/backward swing needs high strength
            'knee': 100.0,   # Knees support the body weight
            'ankle': 20.0    # Ankles should be soft to absorb ground impact
        }
        damping = {
            'hip_x': 1.0,
            'hip_z': 1.0,
            'hip_y': 2.0,
            'knee': 2.0,
            'ankle': 0.5
        }
        action_scale = 0.5