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
        scene_type=None,
        curriculum_mode=None,
        terrain_type=None,
        use_camera=None,
        use_privileged=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            scene_type=scene_type,
            curriculum_mode=curriculum_mode,
            terrain_type=terrain_type,
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
        self.g_x = 0.0                      # World Z-axis projected on torso's local X-axis
        self.g_z = 0.0                      # World Z-axis projected on torso's local Z-axis
        self.lin_vel_x = 0.0                # Current forward linear velocity (m/s)
        self.lin_vel_y = 0.0                # Current lateral linear velocity (m/s)
        self.lin_vel_z = 0.0                # Current vertical linear velocity (m/s)
        self.ang_vel_z = 0.0                # Current angular velocity around world Z-axis (rad/s)
        self.progress = 0.0                 # Distance traveled forward in the current step (m)
        self.last_x_pos = 0.0               # Global X position of torso in the previous frame (m)
        self.torso_height = 0.0             # Height of torso relative to the local terrain (m)
        self.left_foot_height = 0.0         # Height of left foot relative to the local terrain (m)
        self.right_foot_height = 0.0        # Height of right foot relative to the local terrain (m)
        self.left_hip_z = 0.0               # Current angle of the left hip_z (yaw) joint (rad)
        self.right_hip_z = 0.0              # Current angle of the right hip_z (yaw) joint (rad)
        self.knee_contact = False           # Whether knees are touching ground
        self.left_foot_contact = False      # Whether left foot is touching any surface
        self.right_foot_contact = False     # Whether right foot is touching any surface
        self.left_touchdown = False         # Whether left foot hits ground in the current frame but didn't in the last one
        self.right_touchdown = False        # Whether right foot hits ground in the current frame but didn't in the last one
        self.left_air_time = 0.0            # Time elapsed since the left foot last touched the ground (s)
        self.right_air_time = 0.0           # Time elapsed since the right foot last touched the ground (s)
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
        """
        return 42

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

        proprioception = np.concatenate([
            self.data.qpos.flat[2:7],
            rel_joint_pos,
            self.data.qvel.flat[:6],
            self.data.qvel.flat[6:],
            self.data.sensordata.flat,
            gravity_vector
        ])

        obs_parts = [proprioception]

        # Privileged data (Teacher Mode)
        if hasattr(self.config, 'privileged_info') and self.config.privileged_info.enabled:
            terrain_heights = []
            if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen:
                torso_x, torso_y = self.data.qpos[0], self.data.qpos[1]

                # Get the lowest foot height to act as relative zero
                foot_left_z = self.data.geom('foot_left_geom').xpos[2]
                foot_right_z = self.data.geom('foot_right_geom').xpos[2]
                foot_z = min(foot_left_z, foot_right_z)

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

                        # Return absolute height relative to world z=0
                        ground_z = self.terrain_gen.get_height_at(self.model, self._hfield_id, hx, hy)
                        terrain_heights.append(ground_z)
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
        self.progress = 0.0
        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.left_touchdown = False
        self.right_touchdown = False
        self.stumble_force = 0.0
        return obs

    def _update_states(self, action, torques, observation):
        """
        Extracts and stores physical states needed for computing rewards.
        This function is called every step after the physics simulation.
        """
        self.current_torques = torques
        self.current_action = action

        # Orientation
        self.g_x = observation[39]
        self.g_z = observation[41]

        # Velocities
        self.lin_vel_x, self.lin_vel_y, self.lin_vel_z = self.data.qvel[0:3]
        self.ang_vel_z = self.data.qvel[5]

        # Position & progress
        torso_x, torso_y, torso_z = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        # Distance moved since the previous frame
        self.progress = torso_x - self.last_x_pos
        self.last_x_pos = torso_x

        # Heights
        terrain_height = self.terrain_gen.get_height_at(self.model, self._hfield_id, torso_x, torso_y)\
            if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen else 0.0
        self.torso_height = torso_z - terrain_height

        left_foot_pos = self.data.geom('foot_left_geom').xpos
        right_foot_pos = self.data.geom('foot_right_geom').xpos
        left_ground_z = self.terrain_gen.get_height_at(
            self.model, self._hfield_id, left_foot_pos[0], left_foot_pos[1]) if self.terrain_gen else 0.0
        right_ground_z = self.terrain_gen.get_height_at(
            self.model, self._hfield_id, right_foot_pos[0], right_foot_pos[1]) if self.terrain_gen else 0.0
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

        # Air time tracking
        if not self.left_foot_contact:
            self.left_air_time += self.dt  # Increment timer if foot is in the air

        if not self.right_foot_contact:
            self.right_air_time += self.dt

        self.left_touchdown = self.left_foot_contact and not prev_left_contact
        self.right_touchdown = self.right_foot_contact and not prev_right_contact

        # Foot linear velocity in X/Y plane (m/s), used for foot slip penalty
        foot_left_body_id = self.model.body('foot_left').id
        foot_right_body_id = self.model.body('foot_right').id
        self.left_foot_vel = self.data.cvel[foot_left_body_id][3:5]
        self.right_foot_vel = self.data.cvel[foot_right_body_id][3:5]

        # Stumble detection (Friction cone)
        self.stumble_force = 0.0
        foot_left_geom_id = self.model.geom('foot_left_geom').id
        foot_right_geom_id = self.model.geom('foot_right_geom').id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in [foot_left_geom_id, foot_right_geom_id] or contact.geom2 in [foot_left_geom_id, foot_right_geom_id]:
                # Get the contact force vector
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)

                # Transform to world coordinates to separate Horizontal vs Vertical force
                frame = contact.frame.reshape(3, 3)
                f_world = frame.T @ c_array[:3]

                f_xy = np.linalg.norm(f_world[:2])  # Horizontal impact
                f_z = abs(f_world[2])               # Vertical load

                # Penalty if horizontal force is > 2x the vertical load (indicates kicking a wall)
                if f_xy > 2.0 * f_z:
                    self.stumble_force += f_xy

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
        """Rewards keeping the torso perfectly vertical and penalizes leaning backward"""
        # Upright bonus: g_z = 1.0 => vertical; 0.0 => horizontal
        reward = 1.0 if self.g_z > 0.9 else (self.g_z if self.g_z > 0 else 0.0)

        # Penalizes leaning too far forward (< -0.1) or backward (> 0.1)
        if abs(self.g_x) > 0.1:
            reward -= 2.0 * abs(self.g_x)
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

    def _reward_progress(self):
        """Rewards the agent for absolute distance traveled along the X-axis"""
        return self.progress

    def _reward_alive(self):
        """Rewards staying upright and above the ground"""
        return 1.0 if (self.torso_height > self.config.rewards.base_height_target) else 0.0

    def _reward_hip_yaw(self):
        """Penalizes twisting the hip_z joints (duck-footed stance)"""
        return np.square(self.left_hip_z) + np.square(self.right_hip_z)

    def _reward_knee_contact(self):
        """Penalizes agent for knee-walking"""
        return 1.0 if self.knee_contact else 0.0

    def _reward_air_time(self):
        """Rewards holding the foot in the air for an optimal duration before stepping"""
        reward = 0.0
        target_air_time = 0.5

        if self.left_touchdown:
            reward += np.exp(-np.square(self.left_air_time - target_air_time) / 0.1)
            self.left_air_time = 0.0

        if self.right_touchdown:
            reward += np.exp(-np.square(self.right_air_time - target_air_time) / 0.1)
            self.right_air_time = 0.0
        return reward

    def _reward_symmetry(self):
        """Rewards alternating feet (left, then right)"""
        return 1.0 if (self.left_foot_contact ^ self.right_foot_contact) else 0.0

    def _reward_swing(self):
        """Rewards lifting a foot near the ideal target height during a step"""
        reward = 0.0
        if not self.left_foot_contact and self.right_foot_contact:
            error = self.left_foot_height - self.config.rewards.swing_height_target
            reward += np.exp(-np.square(error) / 0.01)
        if not self.right_foot_contact and self.left_foot_contact:
            error = self.right_foot_height - self.config.rewards.swing_height_target
            reward += np.exp(-np.square(error) / 0.01)
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
