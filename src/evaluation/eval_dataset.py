import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

from eval_constants import *
from eval_model import plot_curve
from src.dump import pickle_dump
import skvideo.measure as measure
import cv2
from src.prep.utils import get_clip_paths
import lpips as lp
import torch
import torchvision.transforms as transforms
pd.set_option('use_inf_as_na',True)

loss_fn_alex = lp.LPIPS(net='alex')


def snapshot(path: str, sample_ratio: int = 4):
    arr = [v for i, v in enumerate(sorted(glob.glob(path))) if i % sample_ratio == 0]
    size = (5, 5)
    dirname = os.path.dirname(path).split('/')[:-2]
    restdirname = os.path.dirname(path).split('/')[-2:]

    fig = plt.figure(figsize=size, dpi=300, layout='tight')
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0, aspect='equal', share_all=True, label_mode='1')

    for ax, clip_path in zip(grid, arr):
        ax.axis('off')
        ax.imshow(Image.open(clip_path).crop((280, 0, 1000, 720)))
    print(f'{dirname}/sample.png')
    fig.savefig(f'{os.path.join(*dirname)}/{"_".join(restdirname)}_snapshot.png', bbox_inches='tight')


def stat_dataset(pkl: str):
    df: pd.DataFrame = pickle.load(open(pkl, "rb")).round(2)
    fps_counts = df['fps'].value_counts()
    colors = 0.9 - np.random.rand(len(fps_counts), 3, ) / 2
    _ = plt.figure('STM FPS count', figsize=(1920 / 300, 1080 / 300), dpi=300, layout='tight')
    plt.scatter(fps_counts.index, fps_counts.values, s=fps_counts.values, c=colors)
    plt.yscale('log')
    plt.ylim(5e-1, 1e4)
    plt.ylabel('STM Dataset Clip Count')
    plt.xlabel('Frame Rate (FPS)')
    plt.tight_layout()
    plt.show()


def load_video(path: str, ext: str = 'png', colored: bool = False) -> np.ndarray:
    frames = glob.glob(f'{path}/*.{ext}')
    if colored:
        return np.array([
            cv2.normalize(
                cv2.cvtColor(
                    cv2.resize(
                        cv2.imread(frame, cv2.IMREAD_COLOR),
                        (1280, 720)
                    ),
                    cv2.COLOR_BGR2RGB
                ).astype('float32'),
                None, -1, 1, cv2.NORM_MINMAX
            )
            for frame in frames
        ])
    else:
        return np.array([cv2.resize(cv2.imread(frame, cv2.IMREAD_GRAYSCALE), (1280, 720)) for frame in frames])


def load_videos(paths: list, ext: str = 'png', colored: bool = False) -> list:
    return [load_video(path, ext, colored) for path in paths]


def _call(metric, video: np.ndarray, ref_video: np.ndarray = None) -> np.ndarray:
    if ref_video is not None and metric.__name__ == 'niqe':
        return metric(ref_video)
    elif ref_video is not None:
        return metric(video, ref_video)
    else:
        return metric(video)


def eval_ds(root: str, clip_ids: np.ndarray, metric, par: str, fmt: str, colored: bool = False):
    if not os.path.exists(root):
        raise ValueError(f'root {root} does not exist')
    if not os.path.exists(f'{root}/{par}'):
        raise ValueError(f'partition {root}/{par} does not exist')

    GT_SET = 'gt'
    for ref_set in ['lq', 'edvr', 'realbasicvsr', 'basicvsr', 'basicvsrpp', 'ganbasicvsr']:
        for idx, clip in enumerate(clip_ids):
            print(f'{ref_set} {idx}/{len(clip_ids)}, clip:{clip}')

            gt, lq = load_videos(get_clip_paths(root, par, clip, sets=[GT_SET, ref_set], fmt=fmt), colored=colored)
            assert gt.shape[0] != 0, f'gt shape {gt.shape}'
            assert lq.shape[0] != 0, f'lq shape {lq.shape}'

            res = _call(metric, gt, lq)
            pickle_dump(res, f'{root}/stats/{par}/{ref_set}/{metric.__name__}_{clip:04d}.pkl')


def lpips(img0: np.ndarray, img1: np.ndarray):
    tensor0 = torch.from_numpy(img0).float()
    tensor1 = torch.from_numpy(img1).float()
    tensor = torch.stack((tensor0, tensor1))
    tensor = torch.transpose(tensor, 2, 4)
    tensor = torch.transpose(tensor, 3, 4)
    d = loss_fn_alex(tensor[0], tensor[1])
    return d.detach().numpy().flatten().tolist()


def plot_metric_by_dataset(root_paths: list, metric_name: str, ylim: tuple):
    fig, ax = plt.subplots(figsize=(8, 6))

    _colors = ['r', 'b', 'g', 'y']
    plots = []
    means = []
    for i, root in enumerate(root_paths):
        pkl_files = glob.glob(f'{root}/stats/{metric_name}_*')
        collected_results = [pickle.load(open(pkl, 'rb')) for pkl in pkl_files]
        dataset_stat = pd.DataFrame(collected_results, index=pkl_files).transpose()
        means.append(dataset_stat.iloc[:, :].mean().mean())
        dataset_stat = dataset_stat.iloc[:, :30]
        boxplot = ax.boxplot(dataset_stat, vert=True, patch_artist=True, labels=np.arange(1, 31), widths=0.8,
                             flierprops={'marker': 'x', 'markersize': 2})
        for box in boxplot['boxes']:
            box.set(linewidth=0)
            box.set_facecolor(colors.to_rgba(_colors[i], 0.3))
        for median in boxplot['medians']:
            median.set_color(_colors[i])
        plots.append(boxplot)
    ax.legend([plots[0]["boxes"][0], plots[1]["boxes"][0]], root_paths, loc='lower right')
    ax.set_xlim((0, 31))
    ax.set_ylim(ylim)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xlabel('Random Sequences From Each Dataset')
    ax.set_ylabel(f'{metric_name.upper()} Score')
    fig.tight_layout()
    plt.show()
    return
    fig.savefig(f'data/{metric_name}_comp.png', dpi=300)

    for p, mean in zip(root_paths, means):
        print(f'{p:20s} {metric_name}: {mean}')


def plot_metric_by_model(root_paths: list, metric_name: str, ylim: tuple):
    fig, ax = plt.subplots(figsize=(8, 6))

    _colors = ['r', 'b', 'g', 'c', 'm', 'y']
    plots = []
    means = []
    for i, root in enumerate(root_paths):
        pkl_files = glob.glob(f'data/STM3k/stats/test30/{root}/{metric_name}_*')
        assert len(pkl_files) > 0, f'no pkl files found in data/STM3k/stats/test30/{root}/{metric_name}_*'

        collected_results = [pickle.load(open(pkl, 'rb')) for pkl in pkl_files]
        dataset_stat = pd.DataFrame(collected_results, index=pkl_files).transpose()
        means.append(dataset_stat.iloc[:, :].mean().mean())
        dataset_stat = dataset_stat.iloc[:, :30]
        boxplot = ax.boxplot(dataset_stat, vert=True, patch_artist=True, labels=np.arange(1, 31), widths=0.8,
                             flierprops={'marker': 'x', 'markersize': 2})
        for box in boxplot['boxes']:
            box.set(linewidth=0)
            box.set_facecolor(colors.to_rgba(_colors[i], 0.3))
        for median in boxplot['medians']:
            median.set_color(_colors[i])
        plots.append(boxplot)

    legends = [p["boxes"][0] for p in plots]
    if metric_name == 'lpips':
        ax.legend(legends, root_paths, loc='upper right')
    else:
        ax.legend(legends, root_paths, loc='lower right')
    ax.set_xlim((0, 31))
    ax.set_ylim(ylim)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xlabel('Random Sequences From Each Dataset')
    ax.set_ylabel(f'{metric_name.upper()} Score')
    fig.tight_layout()
    fig.savefig(f'data/{metric_name}_comp.png', dpi=300)

    for p, mean in zip(root_paths, means):
        print(f'{p:20s} {metric_name}: {mean}')


if __name__ == "__main__":
    # snapshot('data/STM/val/gt/0000/*')
    # stat_dataset(f"{DATA_ROOT}cat_stats.pkl")
    # np.random.seed(3407)
    '''eval_ds('data/STM', np.random.randint(0, 300, 31), lpips, 'val',
            fmt='{dataset}/{partition}/{set}/{clip_id:04}', colored=True)
    eval_ds('data/REDS', np.arange(0, 30), lpips, 'val',
            fmt='{dataset}/{partition}/{set}/{clip_id:03}', colored=True)'''
    '''eval_ds('data/STM3k', np.arange(0, 30), measure.psnr, 'test30',
            fmt='{dataset}/{partition}/{set}/{clip_id:04}', colored=False)
    eval_ds('data/STM3k', np.arange(0, 30), measure.ssim, 'test30',
            fmt='{dataset}/{partition}/{set}/{clip_id:04}', colored=False)'''
    '''eval_ds('data/STM3k', np.arange(0, 30), measure.niqe, 'test30',
            fmt='{dataset}/{partition}/{set}/{clip_id:04}', colored=False)'''
    '''eval_ds('data/STM3k', np.arange(0, 30), lpips, 'test30',
            fmt='{dataset}/{partition}/{set}/{clip_id:04}', colored=True)'''
    #plot_metric_by_dataset(['data/REDS', 'data/STM'], 'niqe', (0, 30))
    #plot_metric_by_dataset(['data/REDS', 'data/STM'], 'ssim', (0.8, 1))
    #plot_metric_by_dataset(['data/REDS', 'data/STM'], 'psnr', (10, 50))
    #plot_metric_by_dataset(['data/REDS', 'data/STM'], 'lpips', (0, 1))
    plot_metric_by_model(['lq', 'edvr', 'basicvsr', 'basicvsrpp', 'realbasicvsr', 'ganbasicvsr'], 'psnr', (10, 50))
    #plot_metric_by_model(['lq', 'edvr', 'basicvsr', 'basicvsrpp', 'realbasicvsr', 'ganbasicvsr'], 'ssim', (0.8, 1))
    #plot_metric_by_model(['lq', 'edvr', 'basicvsr', 'basicvsrpp', 'realbasicvsr', 'ganbasicvsr'], 'niqe', (0, 30))
    #plot_metric_by_model(['lq', 'edvr', 'basicvsr', 'basicvsrpp', 'realbasicvsr', 'ganbasicvsr'], 'lpips', (0, 1))