import numpy as np


class BaseConfig:
    pass

class BaseLeggedConfig(BaseConfig):
    """Shared config for all legged robots"""

    class init_state:
        """Defines the starting pose of the robot"""
        pos = [0.0, 0.0, 0.0]           # starting torso position (m)
        default_joint_angles = None

    class asset:
        """Defines the physical model files used by the env"""
        file_name = None

    class control:
        """Defines how the actions are translated into physical forces"""
        stiffness = {}                  # Kp, defines how hard to push to reach the target angle
        damping = {}                    # Kd, defines how much to slow down as it approaches the target to prevent vibration
        action_scale = 1.0              # limits the maximum reach of the neural network

    class domain_rand:
        """Domain randomization settings"""
        push_robots = True              # Enable/disable random pushing during training
        push_interval_steps = 100       # How often to apply the push (steps)
        max_push_force = 50.0           # The maximum random force applied to the torso (N)

    class terrain:
        """Terrain generation settings"""
        enabled = True                  # Toggle terrain generation on/off
        num_levels = 5
        num_lane_cells = 5
        terrain_types = [
            "rough", "stairs_up", "stairs_down", "hill", "trench"
        ]

        # Physical Dimensions (m) - Treated as maximums scaled by difficulty fraction
        base_height = 3.0               # Base height of the floor (to allow pits/trenches)
        roughness = 0.1                 # Maximum height of random bumps
        step_height = 0.2               # Height of each stair step
        step_width = 0.4                # Depth of each stair step
        hill_height = 0.5
        
        # Grid settings
        use_grid = False
        grid_size = 4.0
        num_grid_rows = 5
        num_grid_cols = 5
        grid_feature_margin = 0.1
        grid_terrain_types = ["pyramid", "inv_pyramid", "mound", "bowl", "rough", "flat"]

        # Curriculum settings
        curriculum_promote_fraction = 0.65
        curriculum_demote_fraction = 0.3

        # Other settings
        noise_scale = 0.2               # Downsampling ratio for roughness (higher = more dense)
        feature_margin = 0.4            # Margin before features start (40% of zone size)

    class sensor:

        class depth_camera:
            enabled = False
            width = 16
            height = 16
            near = 0.1               
            far = 8.0                
            update_interval = 5

    class privileged_info:
        """Data given to the Teacher policy"""
        enabled = False
        scan_points_x = np.linspace(-0.5, 1.2, 7)  # Default forward points (m)
        scan_points_y = np.linspace(-0.6, 0.6, 7)  # Default lateral points (m)

    class rewards:
        """Reward settings"""
        class scales:
            pass


class BaseManipulatorConfig(BaseConfig):
    """Shared config for all manipulators"""
