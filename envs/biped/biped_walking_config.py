from envs.biped.biped_config import BipedConfig

class BipedWalkingConfig(BipedConfig):

    class rewards(BipedConfig.rewards):
        class scales:
            progress = 1.0             # Reward for absolute distance traveled along X-axis
            tracking_lin_vel = 1.0     

            # Form and Posture (Prevent "crab walking" or twisting)
            drift = -2.0               # Penalty for drifting sideways
            twist = -2.0               # Penalty for twisting the torso (yaw)
            hip_yaw = -1.0             # Penalty for twisting the hips (duck-footed stance)

            # Behavioral shaping: Prevent "cheating"
            stall = -3.0               # Penalty for standing still to farm survival points
            double_support = -1.0      # Penalty for keeping both feet glued to the ground (shuffling)
            airborne = -5.0            # Penalty for having both feet off the ground (jumping/falling)

            # Stability and Form
            upright = 1.0              # Bonus for keeping the torso perfectly vertical
            air_time = 2.0             # Bonus for a beautiful walking gait (0.5s steps)
            symmetry = 2.0             # Bonus for alternating feet (left, then right)
            swing = 1.0                # Bonus for lifting a foot high into the air

            # Advanced Physics (Crucial for complex terrain)
            foot_slip = -0.01          # Penalty if the foot moves horizontally while touching the ground
            stumble = 0.0              # Penalty if the foot hits a vertical obstacle (like the lip of a stair)
            bounce = -0.1              # Penalty for bouncing/jumping too much vertically

            # Safety and Efficiency
            energy = -0.0001           # Penalty for using massive amounts of torque (saves battery in real life)
            action_rate = -0.01        # Penalty for vibrating/jittering between steps (saves motors in real life)
            knee_contact = -5.0        # Massive penalty if the knees smash into the floor

            # Base survival
            alive = 2.0                # Bonus for staying upright and above the ground

        # Physical thresholds used to calculate rewards and episode termination
        base_height_target = 0.75      # Minimum height to survive; below this, the robot is considered "fallen"
        max_height = 1.5               # Maximum height; above this, the robot probably "exploded" in physics
        min_tilt = 0.5                 # If the gravity vector's Z component drops below 0.5, it tilted too far
        tracking_target_vel = 1.2      # The ideal walking speed the velocity reward pushes towards
        swing_height_target = 0.12     # The ideal height for the feet during swing phase

class BipedTeacherConfig(BipedWalkingConfig):
    class sensor(BipedWalkingConfig.sensor):
        class depth_camera(BipedWalkingConfig.sensor.depth_camera):
            enabled = False

    class privileged_info(BipedWalkingConfig.privileged_info):
        enabled = True
        scan_points_x = [0.1, 0.3, 0.5]
        scan_points_y = [-0.2, 0.0, 0.2]

    class rewards(BipedWalkingConfig.rewards):
        class scales(BipedWalkingConfig.rewards.scales):
            progress = 2.0
            tracking_lin_vel = 2.0
            upright = 1.5
            air_time = 5.0
            alive = 2.0
            stall = -1.0
            stumble = -0.01
            bounce = -0.5
            knee_contact = -10.0

class BipedVisionConfig(BipedWalkingConfig):
    class sensor(BipedWalkingConfig.sensor):
        class depth_camera(BipedWalkingConfig.sensor.depth_camera):
            enabled = True

    class privileged_info(BipedWalkingConfig.privileged_info):
        enabled = False
