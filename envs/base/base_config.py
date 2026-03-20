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
        scene_type = "arena"            # Default scene
        curriculum_mode = "grid"        # Default mode
        nx_zones = 5                    # Number of zones along X-axis
        ny_zones = 5                    # Number of zones along Y-axis
        spawn_zone = (2, 2)             # Default spawn zone
        terrain_types = [
            "rough", "stairs_up", "stairs_down", "hill", "pit"
        ]                               # Types to randomly spawn in grid mode
        single_terrain_type = "flat"    # Default type for single mode

        # Physical Dimensions (m)
        base_height = 0.5               # Base height of the floor
        roughness = 0.05                # Maximum height of random bumps
        step_height = 0.2               # Height of each stair step
        step_width_m = 0.4              # Depth of each stair step
        
        # Other settings
        noise_scale = 0.1               # Downsampling ratio for roughness (0.1 = 1/10th resolution)
        feature_margin = 0.1            # Margin before features start (10% of zone size)
        hill_sigma_range = (0.6, 1.0)   # Range for Gaussian sigma (m) - controls hill width
        taper_distance_m = 0.2          # Boundary tapering distance (m)
        texture_amplitude = 0.01        # Hill surface jitter amplitude
        blend_distance_m = 0.6          # Distance for smoothing clearing edges (m)
        stair_step_range = (2, 5)       # Randomization range for number of steps (min, max+1)
        hill_height_range = (0.3, 0.5)  # Randomization range for hill/pit height
        spawn_lock_size_m = 2.5         # Increased: 5m total flat runway (2.5m front/back)
        spawn_lock_blend_m = 0.5        # Blending distance for spawn zone edge (m)
        global_sigma = 0.8              # Global Gaussian filter strength (pixel-wise)

    class sensor:

        class depth_camera:
            enabled = True
            width = 16
            height = 16
            near = 0.1               
            far = 8.0                
            update_interval = 5

    class privileged_info:
        """Data given to the Teacher policy"""
        enabled = False
        scan_points_x = [0.1, 0.3, 0.5]   # Default forward points (m)
        scan_points_y = [-0.2, 0.0, 0.2]  # Default lateral points (m)

    class rewards:
        """Reward settings"""
        class scales:
            pass


class BaseManipulatorConfig(BaseConfig):
    """Shared config for all manipulators"""
