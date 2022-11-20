from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot(data: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
        image_path: str):
    '''
    Plotting function
    '''
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(image_path)
    plt.close()
    print(f'Plot saved successful at {image_path}!')