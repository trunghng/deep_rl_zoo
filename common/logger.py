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

from common.mpi_utils import proc_rank, mpi_get_statistics, mpi_print
from common.plot import plot


class Logger:


    def __init__(self,
                log_dir: str=None,
                log_fname: str='progress.txt') -> None:
        """
        :param log_dir: (str) Directory for saving experiment results
        :param log_fname: (str) File for saving experiment results
        """
        if proc_rank() == 0:
            self.log_dir = log_dir if log_dir else f'/tmp/experiments/{str(dt.now())}'
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.log_file = open(osp.join(self.log_dir, log_fname), 'w')
            atexit.register(self.log_file.close)
        else:
            self.log_dir = None
            self.log_file = None
        self.first_row = True
        # dict for saving raw data collected over epochs 
        self.raw_epochs_dict = defaultdict(list)
        # dict for saving processed data (mean, std, max, min) collected over epochs
        self.epochs_dict = defaultdict(list)
        self.current_epoch_dict = dict()


    def set_saver(self, what_to_save) -> None:
        self.model = what_to_save


    def save_config(self, config: Dict) -> None:
        if proc_rank() == 0:
            output = json.dumps(config, separators=(',',':\t'), indent=4)
            print('Experiment config:\n', output)
            with open(osp.join(self.log_dir, 'config.json'), 'w') as out:
                out.write(output)
            self.config = config


    def save_state(self, epoch: int=None, msg: bool=False) -> None:
        if proc_rank() == 0:
            ep_txt = f'-ep{ep}' if epoch else ''
            fname = osp.join(self.log_dir, f'model{ep_txt}.pt')
            torch.save(self.model.state_dict(), fname)
            if msg:
                print(f'Model is saved successfully at {fname}')


    def render(self, action_selection: Callable) -> None:
        """Render experiment result as a video

        :param action_selection: action selection function
        """
        if proc_rank() == 0:
            if 'atari' in self.config and self.config['atari']:
                env = make_atari_env(self.config['env'], render_mode='rgb_array')
            else:
                env = gym.make(self.config['env'], render_mode='rgb_array')
            env = RecordVideo(env, video_folder=self.log_dir, disable_logger=True,\
                video_length=self.config['max_ep_len'])
            observation, _ = env.reset()
            step = 0
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
        self.raw_epochs_dict.clear()
        self.current_epoch_dict.clear()
        self.first_row = False
