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
        scene_type=None,
        curriculum_mode=None,
        terrain_type=None,
        use_camera=None,
        use_privileged=None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)

        if use_camera:
            self.config.sensor.depth_camera.enabled = use_camera
        if use_privileged:
            self.config.privileged_info.enabled = use_privileged
        if scene_type:
            self.config.terrain.scene_type = scene_type
        if curriculum_mode:
            self.config.terrain.curriculum_mode = curriculum_mode
        if terrain_type:
            self.config.terrain.terrain_type = terrain_type

        self.terrain_gen = TerrainGenerator(config=self.config)
        self.num_actions = len(self.config.init_state.default_joint_angles)
        self.last_action = np.zeros(self.num_actions)
        self.step_counter = 0
        self.renderer = None
        self.temp_xml_path = None
        self.current_depth_image = np.zeros((
            self.config.sensor.depth_camera.height,
            self.config.sensor.depth_camera.width
        ))
        self._hfield_id = -1

    def _setup_terrain(self):
        """Initializes terrain-related variables after the model is loaded"""
        self._hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')

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
        scene_type = self.config.terrain.scene_type
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
        observation = self._get_obs()
        self._update_states(action, torques, observation)
        total_reward = self._get_total_reward()
        terminated = self._get_termination()
        info = self._get_info()
        info.update(self.info_rewards)
        self.last_action = action

        return observation, total_reward, terminated, False, info

    def reset_model(self):
        if hasattr(self, 'terrain_gen'):
            self.terrain_gen.update_hfield(self.model, self._hfield_id, self.np_random)
            self._needs_hfield_upload = True

            # Sync the offscreen depth camera renderer immediately
            if self.renderer is not None and self._hfield_id != -1:
                con = getattr(self.renderer, '_mjr_context', None)
                if con:
                    mujoco.mjr_uploadHField(self.model, con, self._hfield_id)

        noise_low = -0.01
        noise_high = 0.01
        spawn_pos = list(self.config.init_state.pos)

        # Dynamically adjust spawn height based on terrain
        if self.config.terrain.enabled and hasattr(self, 'terrain_gen') and self.terrain_gen:
            ground_height = self.terrain_gen.get_height_at(self.model, self._hfield_id, spawn_pos[0], spawn_pos[1])
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
