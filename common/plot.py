import argparse, json, os, os.path as osp
from statistics import mean
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


dir_path = osp.dirname(osp.realpath(__file__))
with open(osp.join(dir_path, 'algos_config.json')) as f:
    global algos_config
    algos_config = json.load(f)


def format_label(label):
    return ' '.join(label.split('-')).capitalize()


def plot(data, algo=None, x_axis='total-env-interacts', y_axis='performance', group=None, smooth=1):
    if algo is not None:
        algo = algos_config[algo]
        assert x_axis in algo['time-variants'], f"Available choices for x-axis are {algo['time-variants']}"
        if y_axis == 'performance':
            if algo['type'] == 'on-policy':
                y_axis = 'average-episode-return'
            elif algo['type'] == 'off-policy':
                y_axis = 'average-test-episode-return'
        else:
            assert y_axis in algo['info-variants'], f"Available choices for y-axis are {algo['info-variants']}"

    if isinstance(data, defaultdict):
        data = [pd.DataFrame.from_dict(data)]

    """
    smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
    """
    if smooth > 1:
        smooth = len(data[0][y_axis]) if len(data[0][y_axis]) < smooth else smooth
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[y_axis])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[y_axis] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set()
    sns.lineplot(data=data, x=x_axis, y=y_axis, hue=group, errorbar='sd')
    if group is not None:
        plt.legend(loc='best').set_draggable(True)
    plt.xlabel(format_label(x_axis))
    plt.ylabel(format_label(y_axis))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)


def get_dataset(log_dir, y_axes):
    dataset = []
    log_file = 'progress.txt'
    config_file = 'config.json'
    for root, _, files in os.walk(log_dir):
        if log_file in files:
            try:
                f = open(osp.join(root, config_file))
                config = json.load(f)
                exp_name = config['exp_name'] if 'exp_name' in config else 'exp'
            except:
                print(f'No file named {config_file}')
            data = pd.read_table(osp.join(root, log_file))
            for y_axis in y_axes:
                if y_axis == 'performance':
                    metric = 'average-episode-return' if 'average-episode-return' in data\
                        else 'average-test-episode-return'
                data.insert(len(data.columns), y_axis, data[metric])
            data.insert(len(data.columns), 'exp_name', exp_name)
            dataset.append(data)
    return dataset


def make_plots(log_dirs, x_axes, y_axes, savedir=None):
    x_axes = x_axes if isinstance(x_axes, list) else [x_axes]
    y_axes = y_axes if isinstance(y_axes, list) else [y_axes]

    datasets = []
    for log_dir in log_dirs:
        datasets += get_dataset(log_dir, y_axes)

    for x_axis in x_axes:
        for y_axis in y_axes:
            plt.figure()
            plot(datasets, x_axis=x_axis, y_axis=y_axis, group='exp_name', smooth=7)
            if savedir is not None:
                savepath = osp.join(savedir, f'{x_axis}-{y_axis}.png')
                plt.savefig(savepath)
                print(f'Plotting result is saved at {osp.abspath(savepath)}.')
            else:
                plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Results plotting')
    parser.add_argument('--log-dirs', type=str, nargs='+', help='Directories for saving log files')
    parser.add_argument('-x', '--x-axis', type=str, nargs='*', choices=['epoch', 'total-env-interacts'],
                        default='total-env-interacts', help='Horizontal axes to plot')
    parser.add_argument('-y', '--y-axis', type=str,  nargs='*', default='performance',
                        help='Vertical axes to plot')
    parser.add_argument('-s', '--savedir', type=str,
                        help='Directory to save plotting results')
    args = parser.parse_args()
    make_plots(args.log_dirs, args.x_axis, args.y_axis, args.savedir)
    