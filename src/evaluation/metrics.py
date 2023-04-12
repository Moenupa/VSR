import itertools

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


class Metrics:

    @staticmethod
    def apply_pairwise(values, func):
        """
        Applies a function to a list of values in a pairwise manner.

        Example:
        [1, 2, 3, 4, 5] -> [func(1, 2), func(2, 3), func(3, 4), func(4, 5)]

        Args:
        @values: list of values
        @func: function to apply to the values

        Returns:
        @generator: generator of the results of the function
        """

        def pairwise(iterable):
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        yield from itertools.chain(itertools.starmap(func, pairwise(values)))

    @staticmethod
    def test_apply_pairwise():
        add = lambda x, y: x + y
        ls = [1, 2, 3, 4, 5]
        res = Metrics.apply_pairwise(ls, add)
        return f'test addition pairwise on {ls} -> {list(res)}'

    @staticmethod
    def psnr(img1: np.ndarray, img2: np.ndarray):
        return cv2.PSNR(img1, img2)

    @staticmethod
    def ssim(img1: np.ndarray, img2: np.ndarray):
        # the last channel is assume to be the color channel, hence -1
        return structural_similarity(img1, img2, channel_axis=-1)


if __name__ == '__main__':
    from constants import DATA_ROOT
    lq_path = f'{DATA_ROOT}lq/{0:04d}/{0:08d}.png'
    gt_path = f'{DATA_ROOT}gt/{0:04d}/{0:08d}.png'
    lq = Image.open(lq_path).resize((1280, 720), resample=None)
    gt = Image.open(gt_path).resize((1280, 720), resample=None)
    lq_arr = np.array(lq)
    gt_arr = np.array(gt)
    print(Metrics.ssim(lq_arr, gt_arr))
