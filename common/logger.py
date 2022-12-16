from os.path import dirname, join, realpath
import sys
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
from common.mpi_utils import proc_rank, mpi_mean_std, print_one, mpi_mean
from collections import defaultdict
import torch


class Logger:


    def __init__(self,
                env: str,
                algo: str,
                model_dir: str='./output/models',
                video_dir: str='./output/videos',
                figure_dir: str='./output/figures'):
        self.basename = f'{algo}-{env}'
        self.model_dir = model_dir
        self.video_dir = video_dir
        # self.vid_path = join(video_dir, f'{basename}.mp4')
        # self.plot_path = join(figure_dir, f'{basename}.png')
        self.record = defaultdict(list)
        self.current_row = dict()


    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.record[key].append(value)


    def log_tabular(self, key, value=None):
        if value is None:
            value = mpi_mean(self.record[key][-1])
            key = 'Avg' + key
        self.current_row[key] = value


    def log(self):
        msg = ''
        values = []
        for key, value in self.current_row.items():
            txt_type = '%d' if type(value) == int else '%.4f'
            msg += key + ': ' + txt_type + '\t'
            values.append(value)
        print_one(msg.strip()%tuple(values))


    def set_saver(self, what_to_save):
        self.saver_elements = what_to_save


    def save_state(self, state_dict, itr=None):
        if proc_rank() == 0:
            iter_txt = itr if itr else ''
            fname = join(self.model_dir, f'{self.basename}{iter_txt}.pth')
            torch.save(self.saver_elements, fname)
