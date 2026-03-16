import time, argparse, json
import os.path as osp
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.wrappers.vector import NormalizeObservation, RecordVideo
import numpy as np


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

    from zoo.single.ddpg import DDPG
    from zoo.single.dqn import DQN
    from zoo.single.ppo import PPO
    from zoo.single.sac import SAC
    from zoo.single.trpo import TRPO
    from zoo.single.vpg import VPG

    algos = {
        'ddpg': DDPG, 'dqn': DQN,
        'ppo': PPO, 'sac': SAC,
        'trpo': TRPO, 'vpg': VPG
    }

    if args.save_video:
        render_mode = 'rgb_array'
    elif args.render:
        render_mode = 'human'
    else:
        render_mode = None
    env_kwargs = {}
    if hasattr(config, 'scene_type'): env_kwargs['scene_type'] = config.scene_type
    if hasattr(config, 'curriculum_mode'): env_kwargs['curriculum_mode'] = config.curriculum_mode
    if hasattr(config, 'terrain_type'): env_kwargs['terrain_type'] = config.terrain_type

    env = gym.make_vec(
        config.env,
        num_envs=1, 
        vectorization_mode='sync',
        render_mode=render_mode,
        **env_kwargs
    )

    if getattr(config, 'norm_obs', False):
        env = NormalizeObservation(env)

    if args.save_video:
        video_dir = osp.join(args.log_dir, 'videos')
        print(f'🎬 Recording video to: {video_dir}')
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    model = algos[config.algo](config)
    model_path = args.model_file if args.model_file else osp.join(log_dir, 'latest.pt')
    print(f'Loading weights from: {model_path}')
    model.load(model_path, env=env)

    returns, eps_len = [], []
    try:
        for ep in range(1, args.eps + 1):
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

            while True:
                action_results = model.select_action(obs, action_only=True, deterministic=True)
                action = action_results[0] if isinstance(action_results, tuple) else action_results
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward[0]
                ep_len += 1

                if args.render and not args.save_video:
                    env.render()
                    render_fps = env.metadata.get("render_fps", 60)
                    time.sleep(1.0 / render_fps)

                if terminated[0] or truncated[0] or ep_len == args.max_ep_len:
                    returns.append(ep_ret)
                    eps_len.append(ep_len)
                    print(f'Ep: {ep}\tReturn: {ep_ret:.4f}\tEpLen: {ep_len}')
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--render', action='store_true',
                       help='Render the agent live')
    group.add_argument('--save-video', action='store_true',
                       help='Save a video of the evaluation')
    args = parser.parse_args()
    test(args)
