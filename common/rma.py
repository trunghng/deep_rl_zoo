import argparse, json, os, subprocess, sys
from os.path import abspath, dirname, exists, join
from types import SimpleNamespace
from typing import Tuple, Any, List, Optional

import gymnasium as gym
from gymnasium.wrappers.vector import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common.utils import get_algo_class, get_base_env


class StudentNetwork(nn.Module):
    """Predicts terrain heights from depth image"""

    def __init__(self, input_shape: Tuple[int, int], output_dim: int) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            self.flatten_size = self.conv_layers(dummy).numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc_layers(x)

class VisionDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        data = np.load(data_path)
        self.images = torch.from_numpy(data['depth_images']).float()
        self.heights = torch.from_numpy(data['terrain_heights']).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.heights[idx]

class FusedPolicy(nn.Module):

    def __init__(self, student: nn.Module, teacher: Any, prop_dim: int, priv_dim: int, img_shape: Tuple[int, int], obs_rms: Any = None) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.prop_dim = prop_dim
        self.priv_dim = priv_dim
        self.img_h, self.img_w = img_shape

        self.obs_mean = None
        self.obs_std = None
        if obs_rms is not None:
            device = next(student.parameters()).device
            self.obs_mean = torch.from_numpy(obs_rms.mean).float().to(device)
            self.obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).float() + 1e-8).to(device)

        for param in self.parameters():
            param.requires_grad = False

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_raw = torch.from_numpy(obs).float().to(device)

            # Extract and normalize proprioception
            prop_raw = obs_raw[:, :self.prop_dim]
            if self.obs_mean is not None:
                p_mean = self.obs_mean[:self.prop_dim]
                p_std = self.obs_std[:self.prop_dim]
                prop_norm = (prop_raw - p_mean) / p_std
            else:
                prop_norm = prop_raw

            # Get vision (unnormalized pixels from indices prop_dim to end)
            pixels = obs_raw[:, self.prop_dim:].reshape(-1, 1, self.img_h, self.img_w)
            predicted_heights = self.student(pixels)

            # Normalize vision output for Teacher
            if self.obs_mean is not None:
                h_mean = self.obs_mean[self.prop_dim : self.prop_dim + self.priv_dim]
                h_std = self.obs_std[self.prop_dim : self.prop_dim + self.priv_dim]
                predicted_heights = (predicted_heights - h_mean) / h_std

            combined_obs = torch.cat([prop_norm, predicted_heights], dim=1)
            return self.teacher.select_action(combined_obs, action_only=True, deterministic=deterministic)


class RMA:

    @staticmethod
    def collect_data(model: Any, env: Any, num_steps: int, dataset_path: str) -> None:
        base_env = get_base_env(env, return_vector=False)
        prop_dim, priv_dim = base_env.prop_dim, base_env.privileged_dim
        depth_images, terrain_heights = [], []
        obs, _ = env.reset()  # (1, prop_dim + priv_dim + cam_w * cam_h)
        print(f"RMA Phase 2: Data Collection (Prop: {prop_dim}, Heights: {priv_dim})...")

        for step in tqdm(range(num_steps), desc="Collecting Data"):
            teacher_obs = obs[:, :prop_dim + priv_dim]  # (1, prop_dim + priv_dim)
            if step % 2 == 0:
                heights = obs[0, prop_dim:prop_dim + priv_dim]  # (priv_dim,)
                images = obs[0, prop_dim + priv_dim:]  # (cam_w * cam_h,)
                img_w = base_env.config.sensor.depth_camera.width
                img_h = base_env.config.sensor.depth_camera.height
                depth_images.append(images.reshape(img_h, img_w))
                terrain_heights.append(heights)

            action = model.select_action(teacher_obs, action_only=True, deterministic=True)
            obs, _, _, _, _ = env.step(action)

        env.close()
        np.savez_compressed(dataset_path, depth_images=np.array(depth_images, dtype=np.float32), 
                            terrain_heights=np.array(terrain_heights, dtype=np.float32))
        print(f"Dataset saved to {dataset_path}.")

    @staticmethod
    def train_student(data_path: str, epochs: int, batch_size: int, lr: float, out_model: str) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = VisionDataset(data_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        img_shape = dataset.images.shape[1:]  # (cam_h, cam_w)
        output_dim = dataset.heights.shape[1]  # priv_dim

        print(f"RMA Phase 3: Training Student (Input: {img_shape}, Output: {output_dim})")
        model = StudentNetwork(input_shape=img_shape, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = nn.MSELoss()

        print(f"Training on {len(dataset)} samples using {device}...")
        for epoch in range(epochs):
            epoch_loss = 0
            for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                preds = model(images)
                loss = loss_func(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")

        torch.save(model.state_dict(), out_model)
        print(f"Model saved to {out_model}")

    @staticmethod
    def run_fusion(teacher_log_dir: str, student_model_path: str, num_steps: int) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(join(teacher_log_dir, 'config.json')) as f:
            config_dict = json.load(f)
        config_dict['test_mode'] = True
        config = SimpleNamespace(**config_dict)

        env_kwargs = {}
        if 'env_config' in config_dict and 'terrain' in config_dict['env_config']:
            terrain_config = config_dict['env_config']['terrain']
            env_kwargs = {
                'scene_type': terrain_config.get('scene_type'),
                'curriculum_mode': terrain_config.get('curriculum_mode'),
                'terrain_type': terrain_config.get('terrain_type')
            }

        env = gym.make_vec(
            config.env,
            num_envs=1,
            vectorization_mode='sync',
            use_camera=True,
            use_privileged=False,
            render_mode="rgb_array",
            **env_kwargs
        )
        base_env = get_base_env(env, return_vector=False)

        video_folder = join(teacher_log_dir, 'videos')
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="rma_fusion",
            episode_trigger=lambda x: True
        )

        teacher = get_algo_class(config.algo)(config)
        checkpoint = teacher.load(join(teacher_log_dir, 'latest.pt'))
        obs_rms = checkpoint.get('obs_rms')

        privileged_info = base_env.config.privileged_info
        priv_dim = len(privileged_info.scan_points_x) * len(privileged_info.scan_points_y)

        img_shape = (base_env.config.sensor.depth_camera.height, base_env.config.sensor.depth_camera.width)
        student = StudentNetwork(input_shape=img_shape, output_dim=priv_dim).to(device)
        student.load_state_dict(torch.load(student_model_path, map_location=device))
        student.eval()

        fused_brain = FusedPolicy(student, teacher, base_env.prop_dim, priv_dim, img_shape, obs_rms=obs_rms).to(device)

        print(f"RMA Phase 4: Deploying robot for {num_steps} steps")
        print(f"🎬 Recording deployment video to {video_folder}...")
        obs, _ = env.reset()
        for _ in tqdm(range(num_steps), desc="Deploying"):
            action = fused_brain.select_action(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMA (Rapid Motor Adaptation) Teacher-Student Management CLI.\n\n"
                    "Workflow Phases:\n"
                    "  1. teacher: Train a privileged Teacher model using hidden terrain info.\n"
                    "  2. collect: Run the Teacher to gather (vision, terrain) data pairs.\n"
                    "  3. student: Train a vision Student model to map raw vision to terrain heights offline.\n"
                    "  4. fusion:  Deploy the robot using Student + Teacher.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--phase', type=str, required=True, 
                        choices=['teacher', 'collect', 'student', 'fusion'],
                        help='Select the active RMA phase.')
    args, unknown_args = parser.parse_known_args()

    if args.phase == 'teacher':
        runner = sys.executable if sys.executable else 'python'
        if not unknown_args:
            print("Usage: python -m run rma --phase teacher <algo> [algo_args]")
            return
        algo, algo_args = unknown_args[0], unknown_args[1:]
        algo_file = join(dirname(dirname(abspath(__file__))), 'zoo', 'single', f'{algo}.py')
        print(f"RMA Phase 1: Training Teacher RL (Privileged Info Enabled)...")
        subprocess.check_call([runner, algo_file] + algo_args + ['--use-privileged'])
    elif args.phase == 'collect':
        collect_parser = argparse.ArgumentParser(description='Phase 2: Data Collection')
        collect_parser.add_argument('--log-dir', type=str, required=True,
                                    help='Path to the directory containing teacher model and config file')
        collect_parser.add_argument('--num-steps', type=int, default=50000,
                                    help='Total simulation steps to record')
        collect_parser.add_argument('--out-file', type=str, default='vision_dataset.npz',
                                    help='Output filename for the compressed dataset (saved inside log-dir)')
        collect_args = collect_parser.parse_args(unknown_args)

        with open(join(collect_args.log_dir, 'config.json')) as f:
            config_dict = json.load(f)

        env_kwargs = {}
        if 'env_config' in config_dict and 'terrain' in config_dict['env_config']:
            terrain_config = config_dict['env_config']['terrain']
            env_kwargs = {
                'scene_type': terrain_config.get('scene_type'),
                'curriculum_mode': terrain_config.get('curriculum_mode'),
                'terrain_type': terrain_config.get('terrain_type')
            }
        config_dict['test_mode'] = True
        config = SimpleNamespace(**config_dict)
        model = get_algo_class(config.algo)(config)
        model.load(join(collect_args.log_dir, 'latest.pt'))

        env = gym.make_vec(
            config.env,
            num_envs=1,
            use_camera=True,
            use_privileged=True,
            **env_kwargs
        )

        dataset_path = join(collect_args.log_dir, collect_args.out_file)
        RMA.collect_data(model, env, collect_args.num_steps, dataset_path)
    elif args.phase == 'student':
        student_parser = argparse.ArgumentParser(description='Phase 3: Student Training')
        student_parser.add_argument('--log-dir', type=str, required=True,
                                    help='Path to the directory containing the dataset')
        student_parser.add_argument('--data-file', type=str, default='vision_dataset.npz',
                                    help='Filename of the .npz dataset')
        student_parser.add_argument('--epochs', type=int, default=50,
                                    help='Number of training epochs')
        student_parser.add_argument('--batch-size', type=int, default=64,
                                    help='Mini-batch size for training')
        student_parser.add_argument('--lr', type=float, default=1e-3,
                                    help='Learning rate')
        student_parser.add_argument('--out-model', type=str, default='student_model.pt',
                                    help='Output filename for the trained student model')
        student_args = student_parser.parse_args(unknown_args)

        data_path = join(student_args.log_dir, student_args.data_file)
        model_path = join(student_args.log_dir, student_args.out_model)
        RMA.train_student(
            data_path,
            student_args.epochs,
            student_args.batch_size,
            student_args.lr,
            model_path
        )
    elif args.phase == 'fusion':
        fusion_parser = argparse.ArgumentParser(description='Phase 4: Policy Fusion Deployment')
        fusion_parser.add_argument('--log-dir', type=str, required=True,
                                    help='Path to the teacher log directory')
        fusion_parser.add_argument('--student-model', type=str, required=True,
                                    help='Path to the trained student model weights')
        fusion_parser.add_argument('--num-steps', type=int, default=2000,
                                    help='Number of simulation steps to run for the test')
        fusion_args = fusion_parser.parse_args(unknown_args)

        RMA.run_fusion(
            fusion_args.log_dir,
            fusion_args.student_model,
            fusion_args.num_steps
        )

if __name__ == '__main__':
    main()
