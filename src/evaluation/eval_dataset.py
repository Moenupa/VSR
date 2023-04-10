import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

from eval_constants import *
from eval_model import plot_curve
import skvideo.measure as measure
import cv2
from src.prep.utils import get_clip_paths


def snapshot(path: str, sample_ratio: 5):
    arr = [v for i, v in enumerate(sorted(glob.glob(path))) if i % sample_ratio == 0]
    print(arr)
    return
    fig = plt.figure(par, figsize=size, dpi=100, layout='tight')
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0, aspect='equal', share_all=True, label_mode='1')
    # print(f'{par}/gt/*', glob.glob(f'{par}/gt/*')[:10])
    clip_paths = np.random.choice(glob.glob(f'{self.root}/{par}/gt/*'), size).flatten()

    for ax, clip_path in zip(grid, clip_paths):
        img_path = f'{clip_path}/{frame_id:08d}.png'
        ax.axis('off')
        ax.imshow(Image.open(img_path).crop((280, 0, 1000, 720)))

    fig.savefig(f'data/{par}_sample.png', bbox_inches='tight')


def stat_dataset(pkl: str):
    df: pd.DataFrame = pickle.load(open(pkl, "rb")).round(2)
    fps_counts = df['fps'].value_counts()
    colors = 0.9 - np.random.rand(len(fps_counts), 3, ) / 2
    fig = plt.figure('STM FPS count', figsize=(1920/300, 1080/300), dpi=300, layout='tight')
    plt.scatter(fps_counts.index, fps_counts.values, s=fps_counts.values, c=colors)
    plt.yscale('log')
    plt.ylim(5e-1, 1e4)
    plt.ylabel('STM Dataset Clip Count')
    plt.xlabel('Frame Rate (FPS)')
    plt.tight_layout()
    plt.show()


def load_video(path: str, ext: str = 'png') -> np.ndarray:
    frames = glob.glob(f'{path}/*.{ext}')
    return np.array([cv2.resize(cv2.imread(frame, cv2.IMREAD_GRAYSCALE), (1280, 720)) for frame in frames])


def load_videos(paths: list, ext: str = 'png') -> list:
    return [load_video(path, ext) for path in paths]


def eval_by_metric(metric, video: np.ndarray, ref_video: np.ndarray = None) -> np.ndarray:
    if ref_video is not None:
        return metric(video, ref_video)
    else:
        return metric(video)


def eval_ds(root: str, indices: np.ndarray, metric, fmt: str):
    if not os.path.exists(root):
        raise ValueError(f'path {root} does not exist')

    # create stats folder
    stats_dir = f'{root}/stats'
    os.makedirs(stats_dir, exist_ok=True)

    for idx, clip in enumerate(indices):
        print(f'{idx}/{len(indices)}, clip:{clip}')
        gt, lq = load_videos(get_clip_paths(root, 'val', clip, fmt=fmt))
        assert gt.shape[0] != 0, f'gt shape {gt.shape}'
        assert lq.shape[0] != 0, f'lq shape {lq.shape}'
        res = eval_by_metric(metric, gt, lq)
        pickle.dump(res, open(f'{stats_dir}/{metric.__name__}_{clip}.pkl', 'wb'))


def compare_curve(root_paths: list, metric_name: str, ylim: tuple):
    fig, ax = plt.subplots(figsize=(8, 6))

    _colors = ['r', 'b']
    plots = []
    for i, root in enumerate(root_paths):
        pkl_files = glob.glob(f'{root}/stats/{metric_name}_*')
        collected_results = [pickle.load(open(pkl, 'rb')) for pkl in pkl_files]
        dataset_stat = pd.DataFrame(collected_results, index=pkl_files).transpose().iloc[:, :30]
        boxplot = ax.boxplot(dataset_stat, vert=True, patch_artist=True, labels=np.arange(1,31), widths=0.8,
                   flierprops={'marker': 'x', 'markersize': 2})
        for box in boxplot['boxes']:
            box.set(linewidth=0)
            box.set_facecolor(colors.to_rgba(_colors[i], 0.3))
        for median in boxplot['medians']:
            median.set_color(_colors[i])
        plots.append(boxplot)
    ax.legend([plots[0]["boxes"][0], plots[1]["boxes"][0]], root_paths, loc='lower right')
    ax.set_xlim((0,31))
    ax.set_ylim(ylim)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xlabel('Random Sequences From Each Dataset')
    ax.set_ylabel(f'{metric_name.upper()} Score')
    fig.tight_layout()
    fig.savefig(f'data/{metric_name}_comp.png', dpi=300)
    return


if __name__ == "__main__":
    # stat_dataset(f"{DATA_ROOT}cat_stats.pkl")
    np.random.seed(3407)
    #eval_ds('data/STM', np.random.randint(0, 300, 31), measure.psnr, fmt='{dataset}/{partition}/{set}/{clip_id:04}')
    #eval_ds('data/REDS', np.arange(0, 30), measure.psnr, fmt='{dataset}/{partition}/{set}/{clip_id:03}')
    compare_curve(['data/REDS', 'data/STM'], 'niqe', (0, 30))
    compare_curve(['data/REDS', 'data/STM'], 'ssim', (0.8, 1))
    compare_curve(['data/REDS', 'data/STM'], 'psnr', (10, 50))
    #compare_curve(['data/REDS', 'data/STM'], 'brisque_features', (0, 30))

