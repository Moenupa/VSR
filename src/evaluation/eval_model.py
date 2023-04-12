import os
import os.path as osp
import pickle
from random import randint

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from eval_constants import *
from evaluation.metrics import Metrics

COLORS_RGB = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def basic_paths(dirnames: list, hints: list = None, fmt: str = '{vid:04}/{fid:08d}.png'):
    if hints is None:
        hints = [i.upper() for i in dirnames]
    return {
        hint: f'{DATA_ROOT}{dirname}/{fmt}'
        for hint, dirname in zip(hints, dirnames)
    }


GT = basic_paths(['gt'], ['Ground-Truth'])
LQ = basic_paths(['lq'], ['Low-Quality'])


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
            ax.plot(coldata, label=colname, color=f'c{j}')
            ax.legend(loc='lower right')
            ax.relim()
            if 'PSNR' in colname:
                ax.set_ylim((20, 80))
            elif 'SSIM' in colname:
                ax.set_ylim((0.7, 1))
            else:
                ax.set_ylim((0, None))

    plt.show()


def extract_features(path, features: list, width: int = 1280, height: int = 720) -> list:
    if not osp.exists(path):
        return [None] * (len(features) + 2)

    img = Image.open(path).resize((width, height), resample=None)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, f in enumerate(features):
        draw.rectangle((f[0], f[1], f[2] - 1, f[3] - 1), outline=COLORS_RGB[i], width=3)
    return [img, img_draw] + [img_draw.crop(f).resize((height, height)) for f in features]


def compare(path_interpreter: dict, clip_id, frame_id: int,
            features: list = None):
    if features is None:
        features = []

    # creating a figure, that is a grid with
    n_cols = len(features) + 1
    n_rows = len(path_interpreter)
    figure, axes = plt.subplots(
        n_rows, n_cols,
        sharex='col', sharey='all',
        width_ratios=[1280] + [720] * (n_cols - 1),
        constrained_layout=True
    )
    axes = axes.reshape((n_rows, n_cols))
    gt = None

    for row, (hint, path_fmt) in enumerate(path_interpreter.items()):
        path = path_fmt.format(vid=clip_id, fid=frame_id)
        # print(f'{hint:10s}: {path}')
        original, *outlined = extract_features(path, features)
        if row == 0:
            gt = np.array(original)
            ssim, psnr = 1, 0
        elif original is not None:
            ssim = Metrics.ssim(gt, np.array(original))
            psnr = Metrics.psnr(gt, np.array(original))
        else:
            ssim, psnr = 0, 0
        for col in range(n_cols):
            ax: Axes = axes[row, col]
            ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            if img := outlined[col]:
                ax.imshow(img)
            if row == 0:
                ax.set_title(f'feature_{col}' if col > 0 else 'original')
            if col == 0:
                if row == 0:
                    ax.set_ylabel(f'{hint}\n')
                else:
                    ax.set_ylabel(f'{hint}\n$\\regular_{{PSNR: {psnr:.2f} dB}}$\n$\\regular_{{SSIM: {ssim:.5f}}}$')

    os.makedirs('data/STM3k/eval', exist_ok=True)
    figure.set_size_inches(12, 16)
    figure.savefig(f'data/STM3k/eval/{clip_id:04d}_{frame_id:02d}.png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    # _ = pickle_unpack('data/STM/test/pred.pkl')
    for c in [4, 5, 8, 12, 17, 27, 28, 29]:
        for f in range(1,10):
            print(f'\rclip: {c:02d}, frame: {f}0', end='')
            compare(
                path_interpreter={
                    **GT, **LQ,
                    **basic_paths([
                        'edvr',
                        'basicvsr',
                        'basicvsrpp',
                        'realbasicvsr',
                        'ganbasicvsr'
                    ]),
                },
                clip_id=c, frame_id=f*10,
                features=[(100, 620, 200, 720), (500, 300, 600, 400), (900, 250, 1000, 350)]
            )
