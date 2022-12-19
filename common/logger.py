import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
from deep_rl_zoo.common.mpi_utils import proc_rank, mpi_mean_std, print_one
from collections import defaultdict
import torch
from gym.wrappers.monitoring import video_recorder


class Logger:


    def __init__(self, env, algo ,model_dir, video_dir, figure_dir):
        self.basename = f'{algo}-{env}'
        self.model_dir = model_dir
        self.video_dir = video_dir
        self.figure_dir = figure_dir
        self.record = defaultdict(list)
        self.current_row = dict()


    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.record[key].append(value)


    def log_tabular(self, key, value=None):
        if value is None:
            value, _ = mpi_mean_std(self.record[key][-1])
            key = 'Avg' + key
        self.current_row[key] = value


    def log(self, msg=None):
        if msg:
            print_one(msg)
        else:
            msg = ''
            values = []
            for key, value in self.current_row.items():
                txt_type = '%d' if type(value) == int else '%.4f'
                msg += key + ': ' + txt_type + '\t'
                values.append(value)
            print_one(msg.strip()%tuple(values))


    def set_saver(self, what_to_save):
        self.model = what_to_save


    def save_state(self, epoch=None, msg=False):
        if proc_rank() == 0:
            ep_txt = f'-ep{epoch}' if epoch else ''
            fname = join(self.model_dir, f'{self.basename}{ep_txt}.pth')
            torch.save(self.model.state_dict, fname)
            if msg:
                print(f'Model is saved successfully at {fname}')


    def render(self, env, msg=False):
        if proc_rank() == 0:
            fname = join(self.video_dir, f'{self.basename}.mp4')
            vr = video_recorder.VideoRecorder(env, path=fname)
            obs = env.reset()
            while True:
                env.render()
                vr.capture_frame()
                action, _, _ = self.model.step(obs)
                obs, reward, terminated, _ = env.step(action)
                if terminated:
                    break
            env.close()
            if msg:
                print(f'Video is renderred successfully at {fname}')
