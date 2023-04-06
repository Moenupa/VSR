import glob
import os
from utils import peek_head
import shutil
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

META_DIR = 'meta'
META_FILE = f'{META_DIR}/vid_list.csv'

np.random.seed(3407)


def dump_arr(arr: np.ndarray, path: str, index: bool = True, header: bool = False):
    """
    dump array to csv file

    Args:
        arr: numpy array
        path: path to save
        index: keep the index column in csv
        header: keep the header row in csv

    Returns:

    """
    print(path, '<-', peek_head(arr, 3))
    if os.path.exists(path):
        shutil.move(path, f'{path}.bak')

    # np.savetxt(path, arr, delimiter=",", fmt='%s')
    pd.DataFrame(arr).to_csv(path, index=index, header=header)


def load_arr(path: str, index_col: int = 0, header: int = None) -> list:
    ret = []
    if os.path.exists(path):
        ret = pd.read_csv(path, index_col=index_col, header=header).values.flatten().tolist()
        print(path, '->', peek_head(ret, 3))

    return ret


def clear_dir(paths: list) -> bool:
    for path in paths:
        if not os.path.exists(path):
            print(f'path not exist, {path}')
            return False
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    return True


class Dataset:

    def __init__(self, root: str):
        self.root = root
        os.chdir(root)
        self.vid_list = []
        if os.path.exists(META_FILE):
            self.vid_list = pd.read_csv(META_FILE, header=None).values.flatten().tolist()
        else:
            self.vid_list = Dataset.get_vid_list(ignore_deprecated=True)

        # print(peek_head(self.vid_list, 11))

    @staticmethod
    def get_vid_list(ignore_deprecated: bool = True) -> list:
        """Get all video ids from lq and gt folders

        Args:
            ignore_deprecated: if False, will check if all frames are valid (runs terribly slow)

        Returns:
            list: video ids
        """
        lq = set(os.path.basename(i) for i in glob.glob(f'lq/*'))
        gt = set(os.path.basename(i) for i in glob.glob(f'gt/*'))

        intersection = gt.intersection(lq)
        assert len(intersection) != 0

        all_vid_list = sorted(intersection)
        print('all retrieved video', peek_head(all_vid_list, 3))

        if ignore_deprecated:
            return all_vid_list

        # validate all video frames
        collected = []
        for idx, vid in enumerate(all_vid_list):
            deprecated = False
            lq_frames = glob.glob(f'lq/{vid}/*')
            gt_frames = glob.glob(f'gt/{vid}/*')
            for f in lq_frames + gt_frames:
                deprecated = (cv2.imread(f) is None)
                if deprecated:
                    break

            if not deprecated:
                collected.append(vid)

            print(f'\r{idx}/{len(all_vid_list)}: {vid} {"deprecated" if deprecated else "valid"}', end='')

    def partition(self, dataset_size: int = -1, val_pct: float = 0.1, test_pct: float = 0.1):
        if dataset_size > 0:
            dataset = np.random.choice(self.vid_list, dataset_size, replace=False)
        else:
            dataset = self.vid_list

        val_thresh = int(len(dataset) * (1 - val_pct - test_pct))
        test_thresh = int(len(dataset) * (1 - test_pct))
        train, val, test = np.split(dataset, [val_thresh, test_thresh])

        os.makedirs(META_DIR, exist_ok=True)
        for par, data in [('train', train), ('val', val), ('test', test)]:
            dump_arr(data, f'{META_DIR}/{par}.csv')

            for dtset in ['lq', 'gt']:
                clear_dir([f'{par}/{dtset}'])

                for idx, vid in enumerate(data):
                    os.symlink(os.path.abspath(f'{dtset}/{vid}'), f'{par}/{dtset}/{idx:04d}', target_is_directory=True)

    def restore_partition(self, dry_run: bool = False, from_backup_file: bool = False):
        for par in ['train', 'val', 'test']:
            if from_backup_file:
                dataset = load_arr(f'{META_DIR}/{par}.csv.bak', index_col=0, header=None)
            else:
                dataset = load_arr(f'{META_DIR}/{par}.csv', index_col=0, header=None)

            if dry_run:
                continue

            if not clear_dir([f'{par}/lq', f'{par}/gt']):
                continue

            for idx, vid in enumerate(dataset):
                if vid not in self.vid_list:
                    print(f'video {vid} not found in dataset')
                    continue

                os.symlink(os.path.abspath(f'lq/{vid}'), f'{par}/lq/{idx:04d}', target_is_directory=True)
                os.symlink(os.path.abspath(f'gt/{vid}'), f'{par}/gt/{idx:04d}', target_is_directory=True)

    def sample(self, size: tuple = (10, 10), frame_id: int = 50, partitions=None):
        if partitions is None:
            partitions = ['train', 'test', 'val']
        for par in partitions:
            fig = plt.figure(par, figsize=size, dpi=100, layout='tight')
            grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0, aspect='equal', share_all=True, label_mode='1')
            # print(f'{par}/gt/*', glob.glob(f'{par}/gt/*')[:10])
            clip_paths = np.random.choice(glob.glob(f'{par}/gt/*'), size).flatten()

            for ax, clip_path in zip(grid, clip_paths):
                img_path = f'{clip_path}/{frame_id:08d}.png'
                ax.axis('off')
                ax.imshow(Image.open(img_path).crop((280, 0, 1000, 720)))

            fig.savefig(f'{par}_sample.png', bbox_inches='tight')


if __name__ == '__main__':
    partition = Dataset('data/STM')
    # partition.partition(dataset_size=3000)
    # partition.restore_partition(dry_run=True)
    partition.sample()
