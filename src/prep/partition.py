import glob
import os
from utils import peek_head
import shutil
import pandas as pd
import random
import numpy as np


class Partition:
    def __init__(self, root: str):
        self.root = root
        os.chdir(root)
        lq = set(os.path.basename(i) for i in glob.glob(os.path.join('lq', '*')))
        gt = set(os.path.basename(i) for i in glob.glob(os.path.join('gt', '*')))

        assert len(lq) != 0

        self.vid_list = sorted(gt.intersection(lq))
        print(peek_head(self.vid_list, 3))

    @staticmethod
    def dump_arr(arr: np.ndarray, path: str):
        print(path, '<-', peek_head(arr, 3))
        if os.path.exists(path):
            shutil.move(path, f'{path}.bak')

        # np.savetxt(path, arr, delimiter=",", fmt='%s')
        pd.DataFrame(arr).to_csv(path, index=True, header=False)

    def partition(self, dataset_size: int = -1, val_pct: float = 0.1, test_pct: float = 0.1, metadir: str = 'meta'):
        if dataset_size > 0:
            dataset = np.random.choice(self.vid_list, dataset_size, replace=False)
        else:
            dataset = self.vid_list

        val_thresh = int(len(dataset) * (1 - val_pct - test_pct))
        test_thresh = int(len(dataset) * (1 - test_pct))
        train, val, test = np.split(dataset, [val_thresh, test_thresh])

        os.makedirs(metadir, exist_ok=True)
        for par, data in [('train', train), ('val', val), ('test', test)]:
            Partition.dump_arr(data, f'{metadir}/{par}.csv')

            shutil.rmtree(f'{par}/lq', ignore_errors=True)
            shutil.rmtree(f'{par}/gt', ignore_errors=True)
            os.makedirs(f'{par}/lq', exist_ok=True)
            os.makedirs(f'{par}/gt', exist_ok=True)

            for idx, vid in enumerate(data):
                os.symlink(os.path.abspath(f'lq/{vid}'), f'{par}/lq/{idx:04d}', target_is_directory=True)
                os.symlink(os.path.abspath(f'gt/{vid}'), f'{par}/gt/{idx:04d}', target_is_directory=True)

    def annotate(self):
        pass


if __name__ == '__main__':
    partition = Partition('data/sample')
    partition.partition(dataset_size=10)

