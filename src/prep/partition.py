import glob
import os
from utils import peek_head
import shutil
import pandas as pd
import numpy as np
import cv2

META_DIR = 'meta'
META_FILE = f'{META_DIR}/vid_list.csv'


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


class Partition:

    def __init__(self, root: str):
        self.root = root
        os.chdir(root)
        self.vid_list = []
        if os.path.exists(META_FILE):
            self.vid_list = pd.read_csv(META_FILE, header=None).values.flatten().tolist()
        else:
            self.vid_list = Partition.get_vid_list(ignore_deprecated=True)

        print(peek_head(self.vid_list, 11))

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

            shutil.rmtree(f'{par}/lq', ignore_errors=True)
            shutil.rmtree(f'{par}/gt', ignore_errors=True)
            os.makedirs(f'{par}/lq', exist_ok=True)
            os.makedirs(f'{par}/gt', exist_ok=True)

            for idx, vid in enumerate(data):
                os.symlink(os.path.abspath(f'lq/{vid}'), f'{par}/lq/{idx:04d}', target_is_directory=True)
                os.symlink(os.path.abspath(f'gt/{vid}'), f'{par}/gt/{idx:04d}', target_is_directory=True)

    def restore_partition(self, dry_run: bool = False):
        for par in ['train', 'val', 'test']:
            dataset = load_arr(f'{META_DIR}/{par}.csv', index_col=0, header=None)
            if dry_run:
                continue

            for idx, vid in enumerate(dataset):
                if vid not in self.vid_list:
                    print(f'video {vid} not found in dataset')
                    continue
                os.symlink(os.path.abspath(f'lq/{vid}'), f'{par}/lq/{idx:04d}', target_is_directory=True)
                os.symlink(os.path.abspath(f'gt/{vid}'), f'{par}/gt/{idx:04d}', target_is_directory=True)


if __name__ == '__main__':
    partition = Partition('data/STM')
    # partition.partition(dataset_size=300)
    partition.restore_partition(dry_run=True)
