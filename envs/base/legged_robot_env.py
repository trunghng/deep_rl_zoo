import os, sys
from os.path import abspath, dirname, join

import gymnasium as gym
import mujoco
import numpy as np

from envs.base.base_env import BaseEnv
from envs.utils.terrain import TerrainGenerator

class LeggedRobotEnv(BaseEnv):

    def __init__(
        self,
        config,
        use_camera=None,
        use_privileged=None,
        lane_terrain_types=None,
        lane_difficulties=None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)

        if use_camera:
            self.config.sensor.depth_camera.enabled = use_camera
        if use_privileged:
            self.config.privileged_info.enabled = use_privileged
        if lane_terrain_types is not None:
            self.config.terrain.lane_terrain_types = lane_terrain_types
        if lane_difficulties is not None:
            self.config.terrain.lane_difficulties = lane_difficulties

        self.num_actions = len(self.config.init_state.default_joint_angles)
        self.last_action = np.zeros(self.num_actions)
        self.step_counter = 0
        self.renderer = None
        self.temp_xml_path = None
        self.current_depth_image = np.zeros((
            self.config.sensor.depth_camera.height,
            self.config.sensor.depth_camera.width
        ))
        
        self.terrain_levels = {t: 0 for t in self.config.terrain.terrain_types}
        self.terrain_row = 0
        self.spawn_x = 0.0
        self.spawn_y = 0.0
        self.last_qpos_x = 0.0
        self.first_reset = True
        self._hfield_id = -1

    def _setup_terrain(self):
        """Initializes terrain-related variables after the model is loaded"""
        self._hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')

        if self.config.terrain.enabled and self._hfield_id != -1:
            self.terrain_gen = TerrainGenerator(self.model, self._hfield_id, self.config)

            if not getattr(self.config, 'test_mode', False):
                self.terrain_gen.generate_arena()
            else:
                self.terrain_gen.generate_lane()
            self.terrain_gen._apply_to_mujoco()

            if self.renderer is not None:
                con = getattr(self.renderer, '_mjr_context', None)
                if con:
                    mujoco.mjr_uploadHField(self.model, con, self._hfield_id)
        else:
            self.terrain_gen = None

    def _setup_renderer(self):
        """Initializes the renderer for depth camera after the model is loaded"""
        if self.config.sensor.depth_camera.enabled:
            try:
                self.renderer = mujoco.Renderer(
                    self.model, 
                    self.config.sensor.depth_camera.height, 
                    self.config.sensor.depth_camera.width
                )
            except Exception as e:
                self.renderer = None
                print(f"\n[MuJoCo Warning] Depth Camera Renderer failed to initialize: {e}")
                print("Running without visual observations.")

    def _get_depth_image(self) -> np.ndarray:
        """Captures and normalizes a depth image from the on-board camera"""
        if self.renderer is None:
            return self.current_depth_image

        if self.step_counter % self.config.sensor.depth_camera.update_interval == 0:
            self.renderer.update_scene(self.data, camera='depth_camera')
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()

            # Normalize depth between [near, far]
            near = self.config.sensor.depth_camera.near
            far = self.config.sensor.depth_camera.far
            depth = np.clip(depth, near, far)
            self.current_depth_image = (depth - near) / (far - near)

        return self.current_depth_image

    def _initialize_simulation(self):
        """Overrides Gymnasium's default file loader to build the model in RAM"""
        robot_xml_file = self.config.asset.file_name
        scene_type = 'lane' if getattr(self.config, 'test_mode', False) else 'arena'
        scene_xml_file = f"{scene_type}.xml"

        subclass_dir = dirname(abspath(sys.modules[self.__module__].__file__))
        robot_path = join(subclass_dir, robot_xml_file)
        assets_dir = join(dirname(dirname(abspath(__file__))), "assets")
        scene_path = join(assets_dir, scene_xml_file)

        combined_xml = f"""
        <mujoco model="{self.__class__.__name__.lower()}_{scene_type}">
            <include file="{robot_xml_file}"/>
            <include file="{scene_xml_file}"/>
        </mujoco>
        """

        with open(robot_path, 'rb') as f: robot_content = f.read()
        with open(scene_path, 'rb') as f: scene_content = f.read()

        virtual_assets = {
            robot_xml_file: robot_content,
            scene_xml_file: scene_content
        }

        self.model = mujoco.MjModel.from_xml_string(combined_xml, assets=virtual_assets)
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def close(self):
        """Clean up resources, including the off-screen renderer, when env is closed"""
        if self.renderer is not None:
            try:
                del self.renderer
                self.renderer = None
            except Exception:
                pass
        super().close()

    def _setup_actuators(self, joint_names):
        self.p_gains = np.zeros(self.num_actions)
        self.d_gains = np.zeros(self.num_actions)
        for i, name in enumerate(joint_names):
            self.p_gains[i] = self.config.control.stiffness[name]
            self.d_gains[i] = self.config.control.damping[name]

        self._qpos_indices = [self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.num_actions)]
        self._qvel_indices = [self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(self.num_actions)]

    def _get_total_reward(self):
        total_reward = 0.0
        self.info_rewards = {}

        reward_scales = {
            k: getattr(self.config.rewards.scales, k) 
            for k in dir(self.config.rewards.scales) 
            if not k.startswith('__') and not callable(getattr(self.config.rewards.scales, k))
        }

        for reward_name, scale in reward_scales.items():
            if scale != 0.0:
                reward_func = getattr(self, f'_reward_{reward_name}', None)
                if reward_func:
                    reward_val = reward_func() * scale
                    total_reward += reward_val
                    self.info_rewards[f'reward_{reward_name}'] = reward_val

        return total_reward

    def step(self, action: np.ndarray):
        # Domain randomization
        self.step_counter += 1
        if self.config.domain_rand.push_robots and (self.step_counter % self.config.domain_rand.push_interval_steps == 0):
            push_force = self.np_random.uniform(
                low=-self.config.domain_rand.max_push_force, 
                high=self.config.domain_rand.max_push_force, 
                size=2
            )
            self.data.xfrc_applied[self.model.body('torso').id, :2] = push_force
        else:
            self.data.xfrc_applied[self.model.body('torso').id, :2] = 0

        # PD Control:  torque = Kp * (target - pos) - Kd * vel
        target_qpos = action * self.config.control.action_scale + self.config.init_state.default_joint_angles
        current_qpos = self.data.qpos[self._qpos_indices]
        current_qvel = self.data.qvel[self._qvel_indices]
        torques = self.p_gains * (target_qpos - current_qpos) - self.d_gains * current_qvel

        # Convert raw torques to MuJoCo [-1, 1] control signals
        motor_gears = self.model.actuator_gear[:, 0]
        ctrl_signal = torques / motor_gears

        self.do_simulation(ctrl_signal, self.frame_skip)
        self.last_qpos_x = self.data.qpos[0]

        observation = self._get_obs()
        self._update_states(action, torques, observation)
        total_reward = self._get_total_reward()
        terminated = self._get_termination()
        info = self._get_info()
        info.update(self.info_rewards)
        self.last_action = action

        return observation, total_reward, terminated, False, info

    def reset_model(self):
        if self.config.terrain.enabled and self.terrain_gen:
            if not getattr(self.config, 'test_mode', False):
                if not self.first_reset:
                    distance_walked = self.last_qpos_x - self.spawn_x
                    percentage_walked = distance_walked / self.terrain_gen.cell_length_x

                    current_terrain_type = self.config.terrain.terrain_types[self.terrain_row]
                    current_level = self.terrain_levels[current_terrain_type]

                    if percentage_walked > self.config.terrain.curriculum_promote_fraction:
                        self.terrain_levels[current_terrain_type] = min(self.config.terrain.num_levels - 1, current_level + 1)
                    elif percentage_walked < self.config.terrain.curriculum_demote_fraction:
                        self.terrain_levels[current_terrain_type] = max(0, current_level - 1)

                self.terrain_row = self.np_random.integers(0, len(self.config.terrain.terrain_types))
                next_terrain_type = self.config.terrain.terrain_types[self.terrain_row]
                next_level = self.terrain_levels[next_terrain_type]

                self.spawn_x, self.spawn_y = self.terrain_gen.get_spawn_location(self.terrain_row, next_level)
                self.first_reset = False
            else:
                self.spawn_x, self.spawn_y = self.terrain_gen.get_spawn_location(0, 0)

        noise_low, noise_high = -0.01, 0.01
        spawn_pos = list(self.config.init_state.pos)

        # Add the teleport offset and find the local ground height
        if self.config.terrain.enabled and self.terrain_gen:
            spawn_pos[0] += self.spawn_x
            spawn_pos[1] += self.spawn_y
            ground_height = self.terrain_gen.get_height_at(spawn_pos[0], spawn_pos[1])
            spawn_pos[2] += ground_height

        base_qpos = np.zeros(self.model.nq)
        base_qpos[0:3] = spawn_pos
        base_qpos[3] = 1.0  # Quaternion (w,x,y,z)=(1,0,0,0)

        if self.config.init_state.default_joint_angles is not None:
            base_qpos[7:] = self.config.init_state.default_joint_angles

        qpos = base_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)
        self.last_action = np.zeros(self.num_actions)
        self.step_counter = 0
        return self._get_obs()

    def render(self):
        """Override render to fix the Phantom Terrain bug by syncing GPU memory"""
        if getattr(self, '_needs_hfield_upload', False):
            self._needs_hfield_upload = False
            
            if self._hfield_id != -1 and hasattr(self, 'mujoco_renderer'):
                # Check for human mode
                viewer = getattr(self.mujoco_renderer, 'viewer', None)
                if viewer:
                    con = getattr(viewer, 'con', None)
                    if con: mujoco.mjr_uploadHField(self.model, con, self._hfield_id)
                
                # Check for rgb mode
                renderer = getattr(self.mujoco_renderer, 'renderer', None)
                if renderer:
                    con = getattr(renderer, '_mjr_context', None)
                    if not con:
                        ctx = getattr(renderer, 'context', None)
                        con = getattr(ctx, 'con', None) if ctx else None
                    if con: mujoco.mjr_uploadHField(self.model, con, self._hfield_id)
                
        return super().render()
