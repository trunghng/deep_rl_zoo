from os.path import abspath, dirname, join

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class Bipedal(MujocoEnv, utils.EzPickle):
    """
    The bipedal consists of: 1 free joint at root (torso) & 6 hinge joints per leg
    - nq = 19:
        + 1 for each hinge joint (to represents the angle of the joint)
        + Free joint: 3 for the (X, Y, Z) position in the world;  4 for the orientation
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

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        xml_path = join(dirname(abspath(__file__)), 'bipedal.xml')
        # Each obs is joint of positions and velocities
        # shape = nq + nv + 5: minus the X, Y-coordinate of the root;
        #                       plus 2 (forces on left and right foot);
        #                       plus 3 (direction of gravity vector in robot's frame)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=4,
            observation_space=observation_space,
            **kwargs,
        )

    def _get_obs(self):
        torso_rotation_matrix = self.data.xmat[self.model.body('torso').id].reshape(3, 3)
        # The gravity vector in the robot's local frame is the 3rd row of the rotation matrix
        gravity_vector = torso_rotation_matrix[2, :]
        return np.concatenate([
            self.data.qpos.flat[2:], # skip X, Y-coordinate of the root
            self.data.qvel.flat,
            self.data.sensordata.flat,
            gravity_vector
        ])

    def step(self, action: float):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        # Get the Z-component of the Gravity Vector (Rotation Matrix element R33)
        # This represents 'Up' in the world frame projected onto the robot's 'Up'
        # qpos(17) + qvel(18) + touch(2) + gravity_vec(3)
        g_z = observation[39]
        forward_vel = self.data.qvel[0]
        height = self.data.qpos[2]

        # R1: Velocity Tracking (Gaussian Kernel)
        # Peak reward at 1.0 m/s; bounded [0, 1]
        r_vel = np.exp(-4.0 * np.square(forward_vel - 1.0))

        # R2: Upright Stability (Cosine Projection)
        # Rewards vertical posture; ignores horizontal tilt direction
        r_upright = np.clip(g_z, 0, 1)

        # R3: Control Effort (Quadratic Regularization)
        # Penalizes high torque/energy waste; promotes smoothness
        r_energy = -0.1 * np.square(action).sum()

        # R4: Healthy Bonus (Survival)
        # Rewards the agent for every timestep it stays upright
        r_alive = 1.0 if (height > 0.6 and height < 1.6) else 0.0

        # 3. Final Reward Summation
        reward = r_vel + r_upright + r_energy + r_alive

        # 4. Episode Termination
        # Stop if robot falls (height < 0.6) or "explodes" (height > 1.6)
        terminated = bool(height < 0.6 or height > 1.6)

        info = {
            "reward_dist": r_vel,             # How well is it tracking speed?
            "reward_upright": r_upright,      # Is it actually balanced?
            "reward_energy": r_energy,        # How much torque is it wasting?
            "reward_alive": r_alive,          # Survival bonus
            "forward_velocity": forward_vel,  # Actual physical speed in m/s
            "torso_height": height,           # Actual height in meters
        }
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -0.01
        noise_high = 0.01

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)
        return self._get_obs()
