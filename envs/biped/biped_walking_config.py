from envs.biped.biped_config import BipedConfig

class BipedWalkingConfig(BipedConfig):

    class rewards(BipedConfig.rewards):
        class scales:
            tracking_lin_vel = 5.0     

            # Form and Posture (Prevent "crab walking" or twisting)
            drift = -2.0               # Penalty for drifting sideways
            twist = -2.0               # Penalty for twisting the torso (yaw)
            hip_yaw = -1.0             # Penalty for twisting the hips (duck-footed stance)

            # Behavioral shaping (Prevent "cheating")
            stall = -1.0               # Penalty for standing still to farm survival points

            # Stability and Form
            upright = 2.0              # Bonus for keeping the torso perfectly vertical
            gait_phase = -1.5          # Penalty for feet not matching the phase clock
            swing = 2.0                # Bonus for lifting a foot high into the air

            # Advanced Physics (Crucial for complex terrain)
            foot_slip = -0.5           # Penalty if the foot moves horizontally while touching the ground
            stumble = -1.0             # Massive penalty for hitting vertical obstacles (stair lips)
            bounce = -0.1              # Penalty for bouncing (too high prevents climbing hills)

            # Safety and Efficiency
            energy = -0.0001           # Penalty for using massive amounts of torque
            action_rate = -0.01        # Penalty for vibrating/jittering between steps
            knee_contact = -10.0       # Massive penalty if the knees smash into the floor

            # Base survival
            alive = 2.0                # Bonus for staying upright and above the ground

        # Physical thresholds used to calculate rewards and episode termination
        base_height_target = 0.75      # Minimum height to survive
        max_height = 1.5               # Maximum height
        min_tilt = 0.5                 # Tilted too far
        tracking_target_vel = 1.2      # Target speed
        swing_height_target = 0.23     # Target foot height to safely clear 0.2m stairs without tripping
