import time, argparse, json
import os.path as osp
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward, RecordVideo
import numpy as np
from tqdm import tqdm

from common.utils import get_algo_class
import envs


def test(args) -> None:
    log_dir = args.log_dir
    print(f'Testing policy from: {log_dir}')

    config_path = osp.join(log_dir, 'config.json')
    with open(config_path) as f:
        config_dict = json.load(f)
        config_dict['resume'] = None
        config_dict['use_wandb'] = False
        config_dict['exp_name'] = None
        config_dict['test_mode'] = True
        config = SimpleNamespace(**config_dict)

    if args.save_video:
        render_mode = 'rgb_array'
    elif args.render:
        render_mode = 'human'
    else:
        render_mode = None
    env_kwargs = {}
    if hasattr(config, 'env_config') and 'terrain' in config.env_config:
        terrain_config = config.env_config['terrain']
        priviledged_config = config.env_config['privileged_info']
        depth_camera_config = config.env_config['sensor']['depth_camera']

        if hasattr(args, 'terrains') and args.terrains is not None:
            terrain_config['lane_terrain_types'] = args.terrains[0] if len(args.terrains) == 1 else args.terrains
        else:
            terrain_config['lane_terrain_types'] = None
            
        if hasattr(args, 'difficulties') and args.difficulties is not None:
            diffs = [float(d) for d in args.difficulties]
            terrain_config['lane_difficulties'] = diffs[0] if len(diffs) == 1 else diffs
        else:
            terrain_config['lane_difficulties'] = None

        env_kwargs = {
            'use_camera': depth_camera_config.get('enabled'),
            'use_privileged': priviledged_config.get('enabled'),
            'use_grid': getattr(args, 'use_grid', False),
            'lane_terrain_types': terrain_config['lane_terrain_types'],
            'lane_difficulties': terrain_config['lane_difficulties']
        }

    env = gym.make_vec(
        config.env,
        num_envs=1, 
        vectorization_mode='sync',
        render_mode=render_mode,
        **env_kwargs
    )

    if getattr(config, 'norm_obs', False):
        env = NormalizeObservation(env)
    if getattr(config, 'norm_rew', False):
        env = NormalizeReward(env)

    if args.save_video:
        video_dir = osp.join(args.log_dir, 'videos')
        print(f'🎬 Recording video to: {video_dir}')
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    algo_class = get_algo_class(config.algo)
    model = algo_class(config)
    model_path = args.model_file if args.model_file else osp.join(log_dir, 'latest.pt')
    print(f'Loading weights from: {model_path}')
    model.load(model_path, env=env)

    returns, eps_len = [], []
    try:
        pbar_eps = tqdm(range(1, args.eps + 1), desc="Episodes")
        for ep in pbar_eps:
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

            pbar_steps = tqdm(total=args.max_ep_len, desc=f"Episode {ep}", leave=False)
            while True:
                action_results = model.select_action(obs, action_only=True, deterministic=True)
                action = action_results[0] if isinstance(action_results, tuple) else action_results
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward[0]
                ep_len += 1
                pbar_steps.update(1)

                if args.render and not args.save_video:
                    env.render()
                    render_fps = env.metadata.get("render_fps", 60)
                    time.sleep(1.0 / render_fps)

                if terminated[0] or truncated[0] or ep_len >= args.max_ep_len:
                    returns.append(ep_ret)
                    eps_len.append(ep_len)
                    pbar_steps.close()
                    tqdm.write(f'Ep: {ep}\tReturn: {ep_ret:.4f}\tEpLen: {ep_len}')
                    break
    finally:
        env.close()

    print("-" * 30)
    print(f'Final Results over {args.eps} episodes:')
    print(f'AvgReturn: {np.mean(returns):.4f} (+/- {np.std(returns):.4f})')
    print(f'AvgEpLen:  {np.mean(eps_len):.4f}')
    print("-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy testing')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Path to the log directory')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Specific .pt file to load (defaults to latest.pt)')
    parser.add_argument('--eps', type=int, default=5,
                       help='Number of episodes to test')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                       help='Maximum length of an episode')
    parser.add_argument('--terrains', nargs='*', default=None,
                       help='List of terrain types or "random"')
    parser.add_argument('--difficulties', nargs='*', default=None,
                       help='List of difficulties (0.0 to 1.0) or "random"')
    parser.add_argument('--use-grid', action='store_true',
                       help='Force use of the grid map')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--render', action='store_true',
                       help='Render the agent live')
    group.add_argument('--save-video', action='store_true',
                       help='Save a video of the evaluation')
    args = parser.parse_args()
    test(args)
