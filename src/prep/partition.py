import glob
import os
from utils import peek_head
import csv


def generate_meta(root: str, write: bool = False):
    lq_path = os.path.join('lq', '*')
    gt_path = os.path.join('gt', '*')

    os.chdir(root)
    lq = set(os.path.basename(i) for i in glob.glob(lq_path))
    gt = set(os.path.basename(i) for i in glob.glob(gt_path))

    # lq and gt contains exactly the same
    assert len(lq) != 0
    assert len(gt.symmetric_difference(lq)) == 0

    metas = sorted(gt)
    print(peek_head(metas, 3))
    if write:
        with open('meta.csv', 'w') as f:
            wr = csv.writer(f)
            for meta in metas:
                wr.writerow([meta])


if __name__ == '__main__':
    generate_meta('data/STM5k', write=True)
