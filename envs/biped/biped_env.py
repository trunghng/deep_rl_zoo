from os.path import abspath, dirname, join

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import numpy as np

from envs.biped.biped_config import BipedConfig
from envs.base.legged_robot_env import LeggedRobotEnv

class Bipedal(LeggedRobotEnv):
    """
    The bipedal consists of: 1 free joint at root (torso) & 6 hinge joints per leg
    - nq = 19:
        + 1 for each hinge joint (to represents the angle of the joint)
        + Free joint: 3 for the (X, Y, Z) position in the world; 4 for the orientation
            (using a Quaternion: w, x, y, z)
    - nv = 18:
        + 1 for each hinge joint (to represents the angular velocity)
        + Free joint:  3 for linear velocity (how fast the body is moving through space);
            3 for angular velocity (how fast the body is rotating/spinning)
    """
    metadata = {
       "render_modes": ["human", "rgb_array"],
       "render_fps": 125,
    }

    def __init__(
        self,
        config=BipedConfig(),
        lane_terrain_types=None,
        lane_difficulties=None,
        use_grid=None,
        use_camera=None,
        use_privileged=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            lane_terrain_types=lane_terrain_types,
            lane_difficulties=lane_difficulties,
            use_grid=use_grid,
            use_camera=use_camera,
            use_privileged=use_privileged,
            **kwargs
        )

        obs_size = self.prop_dim + self.privileged_dim + self.vision_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        subclass_dir = dirname(abspath(__file__))
        xml_path = join(subclass_dir, self.config.asset.file_name)
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=4,
            observation_space=self.observation_space,
            **kwargs,
        )

        self._setup_terrain()
        joint_names = [
            'hip_x', 'hip_z', 'hip_y', 'knee', 'ankle', 'ankle', # left
            'hip_x', 'hip_z', 'hip_y', 'knee', 'ankle', 'ankle'  # right
        ]
        self._setup_actuators(joint_names)
        self._setup_renderer()

        # State variables (updated every step)
        self.current_torques = np.zeros(self.num_actions) # Instantaneous torques applied to all motors
        self.current_action = np.zeros(self.num_actions)  # The most recent target joint angles from the agent
        self.gait_period = 0.6              # Time for a full step cycle (s)
        self.left_phase = 0.0
        self.right_phase = 0.0
        self.g_x = 0.0                      # World Z-axis projected on torso's local X-axis
        self.g_y = 0.0                      # World Z-axis projected on torso's local Y-axis
        self.g_z = 0.0                      # World Z-axis projected on torso's local Z-axis
        self.lin_vel_x = 0.0                # Current forward linear velocity (m/s)
        self.lin_vel_y = 0.0                # Current lateral linear velocity (m/s)
        self.lin_vel_z = 0.0                # Current vertical linear velocity (m/s)
        self.ang_vel_z = 0.0                # Current angular velocity around world Z-axis (rad/s)
        self.last_x_pos = 0.0               # Global X position of torso in the previous frame (m)
        self.torso_height = 0.0             # Height of torso relative to the local terrain (m)
        self.left_foot_height = 0.0         # Height of left foot relative to the local terrain (m)
        self.right_foot_height = 0.0        # Height of right foot relative to the local terrain (m)
        self.left_hip_z = 0.0               # Current angle of the left hip_z (yaw) joint (rad)
        self.right_hip_z = 0.0              # Current angle of the right hip_z (yaw) joint (rad)
        self.knee_contact = False           # Whether knees are touching ground
        self.left_foot_contact = False      # Whether left foot is touching any surface
        self.right_foot_contact = False     # Whether right foot is touching any surface
        self.left_foot_vel = np.zeros(2)    # Linear velocity (X, Y) of the left foot (m/s)
        self.right_foot_vel = np.zeros(2)   # Linear velocity (X, Y) of the right foot (m/s)
        self.stumble_force = 0.0            # Magnitude of horizontal impact force on feet (N)

    @property
    def prop_dim(self):
        """Size of the proprioceptive vector (joints, velocities, orientation)
        [0] Torso Z (1)
        [1:5] Torso orientation (4)
        [5:17] Toint pos relative to default (12)
        [17:23] Torso vel (6)
        [23:35] Joint vel (12)
        [35:39] Touch sensors (4)
        [39:42] Projected gravity vector (3)
        [42:46] Phase clock
        """
        return 46

    @property
    def privileged_dim(self):
        """Size of the terrain height grid (if enabled)"""
        if hasattr(self.config, 'privileged_info') and self.config.privileged_info.enabled:
            return len(self.config.privileged_info.scan_points_x) * len(self.config.privileged_info.scan_points_y)
        return 0

    @property
    def vision_dim(self):
        """Total size of the flattened depth image (if enabled)"""
        if self.config.sensor.depth_camera.enabled:
            return self.config.sensor.depth_camera.width * self.config.sensor.depth_camera.height
        return 0

    def _get_obs(self):
        # Proprioceptive data
        torso_rotation_matrix = self.data.xmat[self.model.body('torso').id].reshape(3, 3)
        gravity_vector = torso_rotation_matrix[2, :]
        rel_joint_pos = self.data.qpos[7:] - self.config.init_state.default_joint_angles
        phase_clock = np.array([
            np.sin(2 * np.pi * self.left_phase),
            np.cos(2 * np.pi * self.left_phase),
            np.sin(2 * np.pi * self.right_phase),
            np.cos(2 * np.pi * self.right_phase)
        ])

        proprioception = np.concatenate([
            self.data.qpos.flat[2:7],
            rel_joint_pos,
            self.data.qvel.flat[:6],
            self.data.qvel.flat[6:],
            self.data.sensordata.flat,
            gravity_vector,
            phase_clock
        ])

        obs_parts = [proprioception]

        # Privileged data (Teacher Mode)
        if hasattr(self.config, 'privileged_info') and self.config.privileged_info.enabled:
            terrain_heights = []
            if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen:
                torso_x, torso_y, torso_z = self.data.qpos[0:3]

                # Calculate yaw from the rotation matrix
                yaw = np.arctan2(torso_rotation_matrix[1, 0], torso_rotation_matrix[0, 0])
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)

                for px in self.config.privileged_info.scan_points_x:
                    for py in self.config.privileged_info.scan_points_y:
                        # Rotate local point to global coordinates
                        global_dx = px * cos_y - py * sin_y
                        global_dy = px * sin_y + py * cos_y

                        hx = torso_x + global_dx
                        hy = torso_y + global_dy

                        ground_z = self.terrain_gen.get_height_at(hx, hy)
                        terrain_heights.append(ground_z - torso_z)
            else:
                num_points = len(self.config.privileged_info.scan_points_x) * len(self.config.privileged_info.scan_points_y)
                terrain_heights = [0.0] * num_points

            obs_parts.append(np.array(terrain_heights, dtype=np.float64))

        # Visual data (Student Mode)
        if self.config.sensor.depth_camera.enabled:
            depth_img = self._get_depth_image().flatten()
            obs_parts.append(depth_img)

        return np.concatenate(obs_parts)

    def reset_model(self):
        obs = super().reset_model()
        self.last_x_pos = self.data.qpos[0]
        self.stumble_force = 0.0
        return obs

    def _update_states(self, action, torques, observation):
        """
        Extracts and stores physical states needed for computing rewards.
        This function is called every step after the physics simulation.
        """
        self.current_torques = torques
        self.current_action = action

        # Calculate global phase from 0.0 to 1.0
        global_phase = (self.data.time % self.gait_period) / self.gait_period
        self.left_phase = global_phase
        self.right_phase = (global_phase + 0.5) % 1.0

        # Orientation
        self.g_x, self.g_y, self.g_z = observation[39:42]

        # Velocities
        self.lin_vel_x, self.lin_vel_y, self.lin_vel_z = self.data.qvel[0:3]
        self.ang_vel_z = self.data.qvel[5]

        # Position
        torso_x, torso_y, torso_z = self.data.qpos[0:3]
        self.last_x_pos = torso_x

        # Heights
        terrain_height = self.terrain_gen.get_height_at(torso_x, torso_y)\
            if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen else 0.0
        self.torso_height = torso_z - terrain_height

        left_foot_pos = self.data.geom('foot_left_geom').xpos
        right_foot_pos = self.data.geom('foot_right_geom').xpos
        left_ground_z = self.terrain_gen.get_height_at(left_foot_pos[0], left_foot_pos[1]) if self.terrain_gen else 0.0
        right_ground_z = self.terrain_gen.get_height_at(right_foot_pos[0], right_foot_pos[1]) if self.terrain_gen else 0.0
        self.left_foot_height = left_foot_pos[2] - left_ground_z
        self.right_foot_height = right_foot_pos[2] - right_ground_z

        # Joint angles
        self.left_hip_z = self.data.qpos[8]
        self.right_hip_z = self.data.qpos[14]

        # Contacts
        touch_data = observation[35:39]
        self.knee_contact = (touch_data[0] > 0 or touch_data[1] > 0)

        prev_left_contact = self.left_foot_contact
        prev_right_contact = self.right_foot_contact

        self.left_foot_contact = touch_data[2] > 0
        self.right_foot_contact = touch_data[3] > 0

        # Foot linear velocity in X/Y plane (m/s), used for foot slip penalty
        foot_left_body_id = self.model.body('foot_left').id
        foot_right_body_id = self.model.body('foot_right').id
        self.left_foot_vel = self.data.cvel[foot_left_body_id][3:5]
        self.right_foot_vel = self.data.cvel[foot_right_body_id][3:5]

        # Stumble detection (Surface Normal Analysis)
        self.stumble_force = 0.0
        foot_left_geom_id = self.model.geom('foot_left_geom').id
        foot_right_geom_id = self.model.geom('foot_right_geom').id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            is_left = (contact.geom1 == foot_left_geom_id) or (contact.geom2 == foot_left_geom_id)
            is_right = (contact.geom1 == foot_right_geom_id) or (contact.geom2 == foot_right_geom_id)

            if is_left or is_right:
                # The contact frame's X-axis is the contact normal vector
                frame = contact.frame.reshape(3, 3)
                normal = frame[0] 

                # Threshold of 0.8 treats any slope steeper than ~37 degrees as a stumbling hazard
                is_sloped_obstacle = abs(normal[2]) < 0.8

                if is_sloped_obstacle:
                    # c_array[0] is the normal force pushing against that sloped surface
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, i, c_array)

                    # Use the world-space horizontal component of this impact force
                    # to measure how hard it halted forward momentum
                    f_world = frame.T @ c_array[:3]
                    impact_horizontal = np.linalg.norm(f_world[:2])

                    # Penalizes only if the foot was supposed to be swinging
                    if is_left and self.left_phase >= 0.5:
                        self.stumble_force += impact_horizontal
                    elif is_right and self.right_phase >= 0.5:
                        self.stumble_force += impact_horizontal

    def _get_termination(self) -> bool:
        return bool(
            self.torso_height < self.config.rewards.base_height_target or
            self.torso_height > self.config.rewards.max_height or
            self.g_z < self.config.rewards.min_tilt or
            self.knee_contact
        )

    def _get_info(self) -> dict:
        return {
            'forward_velocity': self.lin_vel_x,
            'torso_height': self.torso_height,
        }

    def _reward_energy(self):
        """Penalizes the sum of squared torques applied to the motors"""
        return np.sum(np.square(self.current_torques))

    def _reward_action_rate(self):
        """Penalizes vibrating/jittering between steps"""
        return np.sum(np.square(self.current_action - self.last_action))

    def _reward_upright(self):
        """Rewards keeping the torso perfectly vertical and penalizes leaning forward/backward or sideways"""
        # Upright bonus: g_z = 1.0 => vertical; 0.0 => horizontal
        reward = 1.0 if self.g_z > 0.9 else (self.g_z if self.g_z > 0 else 0.0)

        # Penalizes leaning too far forward/backward (g_x) or sideways (g_y)
        total_tilt = np.sqrt(np.square(self.g_x) + np.square(self.g_y))
        if total_tilt > 0.1:
            reward -= 2.0 * total_tilt
        return reward

    def _reward_tracking_lin_vel(self):
        """Rewards the agent for moving forward"""
        return np.clip(self.lin_vel_x, 0, self.config.rewards.tracking_target_vel)

    def _reward_stall(self):
        """Penalizes standing still to farm survival points"""
        return 1.0 if (self.lin_vel_x < 0.3) else 0.0

    def _reward_drift(self):
        """Penalizes sideways movement (drifting)"""
        return np.square(self.lin_vel_y)

    def _reward_bounce(self):
        """Penalizes vertical jumping or hopping movement"""
        return np.square(self.lin_vel_z)

    def _reward_twist(self):
        """Penalizes twisting of the torso (yaw spinning)"""
        return np.square(self.ang_vel_z)

    def _reward_alive(self):
        """Rewards staying upright and above the ground"""
        return 1.0 if (self.torso_height > self.config.rewards.base_height_target) else 0.0

    def _reward_hip_yaw(self):
        """Penalizes twisting the hip_z joints (duck-footed stance)"""
        return np.square(self.left_hip_z) + np.square(self.right_hip_z)

    def _reward_knee_contact(self):
        """Penalizes agent for knee-walking"""
        return 1.0 if self.knee_contact else 0.0

    def _reward_gait_phase(self):
        """Rewards the agent for synchronizing foot contacts with the phase clock"""
        reward = 0.0
        left_desired_contact = 1.0 if self.left_phase < 0.5 else 0.0
        left_actual_contact = 1.0 if self.left_foot_contact else 0.0
        right_desired_contact = 1.0 if self.right_phase < 0.5 else 0.0
        right_actual_contact = 1.0 if self.right_foot_contact else 0.0

        # Penalizes the difference between desired and actual contact states
        reward += np.square(left_desired_contact - left_actual_contact)
        reward += np.square(right_desired_contact - right_actual_contact)
        return reward

    def _reward_swing(self):
        """Rewards lifting the foot to follow a dynamic phase-based target arc"""
        reward = 0.0
        if self.left_phase >= 0.5:
            # Calculate target height along the sine wave
            left_target_height = self.config.rewards.swing_height_target * np.sin(2 * np.pi * (self.left_phase - 0.5))
            # Error is only positive if the actual foot height is below the target arc
            left_error = np.maximum(0.0, left_target_height - self.left_foot_height)
            reward += np.exp(-np.square(left_error) / 0.01)

        if self.right_phase >= 0.5:
            right_target_height = self.config.rewards.swing_height_target * np.sin(2 * np.pi * (self.right_phase - 0.5))
            right_error = np.maximum(0.0, right_target_height - self.right_foot_height)
            reward += np.exp(-np.square(right_error) / 0.01)
        return reward

    def _reward_foot_velocity(self):
        """Rewards the swing foot for moving forward relative to the torso."""
        reward = 0.0
        if self.left_phase >= 0.5:
            rel_vel = self.left_foot_vel[0] - self.lin_vel_x
            reward += np.clip(rel_vel, 0, 1.0)
        if self.right_phase >= 0.5:
            rel_vel = self.right_foot_vel[0] - self.lin_vel_x
            reward += np.clip(rel_vel, 0, 1.0)
        return reward

    def _reward_foot_slip(self):
        """Penalizes if the foot moves horizontally while touching the ground"""
        reward = 0.0
        if self.left_foot_contact:
            reward += np.sum(np.square(self.left_foot_vel))
        if self.right_foot_contact:
            reward += np.sum(np.square(self.right_foot_vel))
        return reward

    def _reward_stumble(self):
        """Penalizes if the foot hits a vertical obstacle (like the lip of a stair)"""
        return self.stumble_force
