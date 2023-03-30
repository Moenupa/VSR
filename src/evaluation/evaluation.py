import pickle
import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt

COLORS = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_unpack(path):

    data = list(next(iter(record.values())) for record in pickle_read(path))
    df = pd.DataFrame(data)

    test_size, metrics = df.shape
    print(f'test_size: {test_size}, metrics: {metrics}')

    # figure, axes = plt.subplots(metrics, 1, sharex='all')
    clip_id = randint(0, test_size // 100 - 1) * 100
    print(f'selected clip_id: {clip_id // 100}')

    '''for i in range(metrics):
        ylim = (0, 1) if df.columns[i] == 'SSIM' else (0, None)
        df.iloc[clip_id:clip_id + 100, i].plot(
            ax=axes[i], legend=True,
            xlim=(clip_id - 10, clip_id + 110),
            ylim=ylim,
            color=colors[i],
        )
        axes[i].legend(loc='lower right')
    plt.show()'''
    plot_curve(
        df.iloc[:, :],
        )


def plot_curve(*args, **kwargs):
    curves = {i: v for i, v in enumerate(args)}
    curves.update(kwargs)
    # first num of rows, then columns
    shape = max(v.shape[-1] for v in curves.values()), len(curves)
    print(f'{shape}')

    figure, axes = plt.subplots(*shape, sharex='col', sharey='row')
    axes = axes.reshape(shape)
    for id_curves, (name, curve) in enumerate(curves.items()):
        for j, (colname, coldata) in enumerate(curve.items()):
            ax = axes[j, id_curves]
            ax.plot(coldata, label=colname, color=COLORS[j % len(COLORS)])
            ax.legend(loc='lower right')
            ax.relim()
            if 'PSNR' in colname:
                ax.set_ylim((20, 80))
            elif 'SSIM' in colname:
                ax.set_ylim((0.7, 1))
            else:
                ax.set_ylim((0, None))

    plt.show()

def compare(path_interpreter: list, clip_id: int, frame_id: int):
    for interpreter in path_interpreter:
        path = interpreter(clip_id, frame_id)
        print(path)

    


if __name__ == '__main__':
    _ = pickle_unpack('data/STM/test/pred.pkl')
