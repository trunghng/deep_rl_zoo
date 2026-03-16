from os.path import abspath, dirname, join

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
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
        **kwargs
    ):
        super().__init__(
            config=config,
            scene_type=scene_type,
            curriculum_mode=curriculum_mode,
            terrain_type=terrain_type,
            **kwargs
        )

        subclass_dir = dirname(abspath(__file__))
        xml_path = self._assemble_modular_xml(subclass_dir)

        # Each obs is joint of positions and velocities
        # [0] Torso Z (1)
        # [1:5] Torso orientation (4)
        # [5:17] Toint pos relative to default (12)
        # [17:23] Torso vel (6)
        # [23:35] Joint vel (12)
        # [35:39] Touch sensors (4)
        prop_shape = (42,)
        if self.config.sensor.depth_camera.enabled:
            # 42 proprioception + dynamic depth pixels (width * height)
            img_size = self.config.sensor.depth_camera.width * self.config.sensor.depth_camera.height
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(42 + img_size,), dtype=np.float64)
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=prop_shape, dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=4,
            observation_space=self.observation_space,
            **kwargs,
        )

        joint_names = [
            'hip_x', 'hip_z', 'hip_y', 'knee', 'ankle', 'ankle', # left
            'hip_x', 'hip_z', 'hip_y', 'knee', 'ankle', 'ankle'  # right
        ]
        self._setup_actuators(joint_names)
        self._setup_renderer()

        self.forward_vel = 0.0
        self.lateral_vel = 0.0
        self.yaw_vel = 0.0
        self.torso_height = 0.0
        self.g_z = 0.0
        self.last_x_pos = 0.0
        self.progress = 0.0
        self.left_foot_contact = False
        self.right_foot_contact = False
        self.knee_contact = False
        self.left_foot_z = 0.0
        self.right_foot_z = 0.0
        self.hip_z_left = 0.0
        self.hip_z_right = 0.0
        self.current_torques = np.zeros(self.num_actions)
        self.current_action = np.zeros(self.num_actions)

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

        # Visual data
        if self.config.sensor.depth_camera.enabled:
            depth_img = self._get_depth_image().flatten()
            return np.concatenate([proprioception, depth_img])
        
        return proprioception

    def reset_model(self):
        obs = super().reset_model()
        self.last_x_pos = self.data.qpos[0]
        self.progress = 0.0
        return obs

    def _update_states(self, action, torques, observation):
        """Extracts and stores physical states needed for computing rewards"""
        self.current_action = action
        self.current_torques = torques

        self.g_z = observation[41] 
        self.forward_vel = self.data.qvel[0]
        self.lateral_vel = self.data.qvel[1]
        self.yaw_vel = self.data.qvel[5]
        self.lin_vel_z = self.data.qvel[2]

        torso_x, torso_y, torso_z = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]

        self.progress = torso_x - self.last_x_pos
        self.last_x_pos = torso_x

        self.terrain_height = 0.0
        if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen:
            self.terrain_height = self.terrain_gen.get_height_at(self.model, torso_x, torso_y)

        self.torso_height = torso_z - self.terrain_height

        self.hip_z_left = self.data.qpos[8]
        self.hip_z_right = self.data.qpos[14]

        touch_data = observation[35:39]
        self.knee_contact = (touch_data[0] > 0 or touch_data[1] > 0)
        self.left_foot_contact = touch_data[2] > 0
        self.right_foot_contact = touch_data[3] > 0

        left_foot_pos = self.data.geom('foot_left_geom').xpos
        right_foot_pos = self.data.geom('foot_right_geom').xpos
        left_ground_z = self.terrain_gen.get_height_at(self.model, left_foot_pos[0], left_foot_pos[1])\
            if self.config.terrain.enabled and self.terrain_gen else 0.0
        right_ground_z = self.terrain_gen.get_height_at(self.model, right_foot_pos[0], right_foot_pos[1])\
            if self.config.terrain.enabled and self.terrain_gen else 0.0

        self.left_foot_z = left_foot_pos[2] - left_ground_z
        self.right_foot_z = right_foot_pos[2] - right_ground_z

        foot_left_body_id = self.model.body('foot_left').id
        foot_right_body_id = self.model.body('foot_right').id

        self.left_foot_v = self.data.cvel[foot_left_body_id][3:5]
        self.right_foot_v = self.data.cvel[foot_right_body_id][3:5]

        # Calculate stumble (horizontal contact forces)
        self.stumble_force = 0.0
        foot_left_geom_id = self.model.geom('foot_left_geom').id
        foot_right_geom_id = self.model.geom('foot_right_geom').id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in [foot_left_geom_id, foot_right_geom_id]\
                or contact.geom2 in [foot_left_geom_id, foot_right_geom_id]:
                # Extract force in contact frame
                c_array = np.zeros(6, dtype=np.float64)
                import mujoco
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                # Penalize large forces in the lateral/friction directions
                # c_array[1] and c_array[2] are friction directions in contact frame, c_array[0] is normal
                self.stumble_force += np.linalg.norm(c_array[1:3])

    def _get_termination(self) -> bool:
        return bool(
            self.torso_height < self.config.rewards.base_height_target or
            self.torso_height > self.config.rewards.max_height or
            self.g_z < self.config.rewards.min_tilt or
            self.knee_contact
        )

    def _get_info(self) -> dict:
        return {
            'forward_velocity': self.forward_vel,
            'torso_height': self.torso_height,
        }

    def _reward_tracking_lin_vel(self):
        """Rewards the agent for moving forward"""
        return np.clip(self.forward_vel, 0, self.config.rewards.tracking_target_vel)

    def _reward_progress(self):
        """Rewards the agent for absolute distance traveled along the X-axis"""
        return self.progress

    def _reward_lin_vel_y(self):
        """Penalizes sideways movement"""
        return np.square(self.lateral_vel)

    def _reward_ang_vel_z(self):
        """Penalizes twisting of the torso (yaw)"""
        return np.square(self.yaw_vel)

    def _reward_hip_yaw(self):
        """Penalizes twisting the hip_z joints (duck-footed stance)"""
        return np.square(self.hip_z_left) + np.square(self.hip_z_right)

    def _reward_stall(self):
        """Penalizes standing still to farm survival points"""
        return 1.0 if (self.forward_vel < 0.3) else 0.0

    def _reward_upright(self):
        """Rewards keeping the torso perfectly vertical"""
        return 1.0 if self.g_z > 0.9 else (self.g_z if self.g_z > 0 else 0.0)

    def _reward_symmetry(self):
        """Rewards alternating feet (left, then right)"""
        return 1.0 if (self.left_foot_contact ^ self.right_foot_contact) else 0.0

    def _reward_swing(self):
        """Rewards lifting a foot near the ideal target height during a step"""
        reward = 0.0
        if not self.left_foot_contact:
            error = self.left_foot_z - self.config.rewards.swing_height_target
            reward += np.exp(-np.square(error) / 0.01)
        if not self.right_foot_contact:
            error = self.right_foot_z - self.config.rewards.swing_height_target
            reward += np.exp(-np.square(error) / 0.01)
        return reward

    def _reward_double_support(self):
        """Penalizes keeping both feet glued to the ground"""
        return 1.0 if (self.left_foot_contact and self.right_foot_contact) else 0.0

    def _reward_airborne(self):
        """Penalizes having both feet off the ground (unstable)"""
        return 1.0 if (not self.left_foot_contact and not self.right_foot_contact) else 0.0

    def _reward_energy(self):
        """Penalizes the sum of squared torques applied to the motors"""
        return np.sum(np.square(self.current_torques))

    def _reward_alive(self):
        """Rewards staying upright and above the ground"""
        return 1.0 if (self.torso_height > self.config.rewards.base_height_target) else 0.0

    def _reward_knee_contact(self):
        """Massive penalty if the knees smash into the floor"""
        return 1.0 if self.knee_contact else 0.0

    def _reward_action_rate(self):
        """Penalizes vibrating/jittering between steps"""
        return np.sum(np.square(self.current_action - self.last_action))

    def _reward_foot_slip(self):
        """Penalizes if the foot moves horizontally while touching the ground"""
        reward = 0.0
        if self.left_foot_contact:
            reward += np.sum(np.square(self.left_foot_v))
        if self.right_foot_contact:
            reward += np.sum(np.square(self.right_foot_v))
        return reward

    def _reward_stumble(self):
        """Penalizes if the foot hits a vertical obstacle (like the lip of a stair)"""
        return self.stumble_force

    def _reward_lin_vel_z(self):
        """Penalizes for bouncing/jumping too much vertically"""
        return np.square(self.lin_vel_z)
