'''
Logging utilities, taken with adapted modification from OpenAI Spinning Up's github
Ref:
[1] https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
'''
import time, atexit, os, json
import os.path as osp
from datetime import datetime as dt
from collections import defaultdict
from typing import Callable, Dict

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from common.mpi_utils import proc_rank, mpi_get_statistics, mpi_print
from common.plot import plot


class Logger:


    def __init__(self,
                log_dir: str=None,
                log_fname: str='progress.txt',
                config=None) -> None:
        self.config = config or {}
        self.use_wandb = config.get('use_wandb', False)
        self.wandb_id = config.get('wandb_id', None)
        self.test_mode = self.config.get('test_mode', False)
        
        if proc_rank() == 0 and not self.test_mode:
            self.log_dir = log_dir if log_dir else f'/tmp/experiments/{str(dt.now())}'
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = osp.join(self.log_dir, log_fname)
            exists = osp.exists(self.log_path) and osp.getsize(self.log_path) > 0
            self.log_file = open(self.log_path, 'a')
            self.first_row = not exists

            if self.use_wandb:
                wandb.init(
                    project='drl_zoo',
                    id=self.wandb_id,
                    resume='allow',
                    config=self.config,
                    name=self.config.get('exp_name', None),
                    monitor_gym=True,
                    save_code=True
                )
        else:
            self.log_dir = None
            self.log_file = None
            self.first_row = True
        # dict for saving raw data collected over epochs 
        self.raw_epochs_dict = defaultdict(list)
        # dict for saving processed data (mean, std, max, min) collected over epochs
        self.epochs_dict = defaultdict(list)
        self.current_epoch_dict = dict()

    def save_config(self, config: Dict, env=None) -> None:
        if proc_rank() == 0 and self.log_dir is not None:
            if env is not None:
                try:
                    config['env_config'] = env.unwrapped.envs[0].unwrapped.get_config()
                except:
                    pass

            output = json.dumps(config, separators=(',',':\t'), indent=4)
            print('Experiment config:\n', output)
            with open(osp.join(self.log_dir, 'config.json'), 'w') as out:
                out.write(output)
            self.config = config

    def save_state(self, state_dict, epoch: int) -> None:
        if proc_rank() == 0 and self.log_dir is not None:
            torch.save(state_dict, osp.join(self.log_dir, f'checkpoint_ep{epoch}.pt'))

    def save_latest(self, state_dict):
        if proc_rank() == 0 and self.log_dir is not None:
            torch.save(state_dict, osp.join(self.log_dir, 'latest.pt'))

    def render(self, action_selection: Callable, video: bool = True) -> None:
        """Render experiment result as a video or live human view
        
        :param action_selection: action selection function
        :param video: If True, saves an mp4. If False, renders to the screen
        """
        if proc_rank() == 0:
            render_mode = 'rgb_array' if video else 'human'
            
            if 'atari' in self.config and self.config['atari']:
                env = make_atari_env(self.config['env'], render_mode=render_mode)
            else:
                env = gym.make(self.config['env'], render_mode=render_mode)
            
            if video:
                env = RecordVideo(env, video_folder=self.log_dir, disable_logger=True,
                                  video_length=self.config['max_ep_len'])
            
            observation, _ = env.reset()
            while True:
                action = action_selection(observation)
                observation, reward, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break
            env.close()

    def plot(self) -> None:
        """Plot experiment results"""
        if proc_rank() == 0:
            plt.figure()
            fname = osp.join(self.log_dir, 'plot.png')
            plot(self.epochs_dict, self.config['algo'], smooth=7)
            plt.savefig(fname)
            plt.close()
            print(f'Plot is saved successfully at {fname}')

    def add(self, data):
        for key, value in data.items():
            self.raw_epochs_dict[key].append(value)

    def log_epoch(self, key, value=None, average_only=False, need_optima=False):
        if value is None:
            values = self.raw_epochs_dict[key]
            values = np.concatenate(values) if isinstance(values[0], np.ndarray)\
                and len(values[0].shape) > 0 else values
            stats = mpi_get_statistics(values, need_optima)
            if average_only:
                self.current_epoch_dict[key] = stats[0]
                self.epochs_dict[key].append(stats[0])
            else:
                self.current_epoch_dict[f'average-{key}'] = stats[0]
                self.current_epoch_dict[f'std-{key}'] = stats[1]
                self.epochs_dict[f'average-{key}'].append(stats[0])
                self.epochs_dict[f'std-{key}'].append(stats[1])
            if need_optima:
                self.current_epoch_dict[f'max-{key}'] = stats[2]
                self.current_epoch_dict[f'min-{key}'] = stats[3]
                self.epochs_dict[f'max-{key}'].append(stats[2])
                self.epochs_dict[f'min-{key}'].append(stats[3])
        else:
            self.current_epoch_dict[key] = value
            self.epochs_dict[key].append(value)

    def log(self, msg):
        mpi_print(msg)

    def dump_epoch(self):
        if proc_rank()==0:
            vals = []
            key_lens = [len(key) for key in self.current_epoch_dict]
            max_key_len = max(15,max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.current_epoch_dict:
                val = self.current_epoch_dict.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.log_file is not None:
                if self.first_row:
                    self.log_file.write("\t".join(self.current_epoch_dict.keys())+"\n")
                self.log_file.write("\t".join(map(str, vals))+ "\n")
                self.log_file.flush()

            if self.use_wandb:
                wandb_logs = {}
                for key, val in self.current_epoch_dict.items():
                    if isinstance(val, (int, float, np.number)):
                        wandb_logs[key] = val
                wandb.log(wandb_logs)
        self.raw_epochs_dict.clear()
        self.current_epoch_dict.clear()
        self.first_row = False

    def truncate_log(self, target_epoch: int):
        """Remove rows from log file that are newer than target_epoch"""
        if proc_rank() != 0 or not self.log_file:
            return

        self.log_file.close()
        lines = []
        with open(self.log_path, 'r') as f:
            all_lines = f.readlines()
            if not all_lines:
                return

            # Keep header
            lines.append(all_lines[0])

            # Keep only lines where epoch <= target_epoch
            for line in all_lines[1:]:
                parts = line.split('\t')
                try:
                    # Assume first column is 'epoch'
                    epoch_val = int(parts[0])
                    if epoch_val <= target_epoch:
                        lines.append(line)
                except:
                    continue

        # Rewrite the file
        with open(self.log_path, 'w') as f:
            f.writelines(lines)

        # Re-open in append mode for training
        self.log_file = open(self.log_path, 'a')

    def close(self):
        if self.log_file:
            self.log_file.close()
