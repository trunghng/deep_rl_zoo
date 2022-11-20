from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot(list_: List[float], xlabel: str, ylabel: str, plot_name: str):
    '''
    Plotting function
    '''
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(list_)), list_)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_name)
    plot_img = plot_name.lower().replace(' ', '_')
    plot_path = f'./outputs/images/{plot_img}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved successful at {plot_path}!')