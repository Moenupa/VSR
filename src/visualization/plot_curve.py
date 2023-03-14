import cv2
import itertools
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import os
import math
from types import FunctionType as function
import numpy as np


class IO():
    get_subdirs = lambda dir: [d.path for d in os.scandir(dir) if d.is_dir()]
    get_paths = lambda regex: sorted(glob.glob(regex))
    get_frames = lambda dir: list(map(cv2.imread, IO.get_paths(f'{dir}/*.png'))
                                  )
    contain_frames = lambda dir: len(glob.glob(f'{dir}/*.png')) > 0


class Metrics():

    @staticmethod
    def apply_pairwise(values, func: function):
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
        return structural_similarity(img1, img2, multichannel=True)


class Curve():
    '''
    Class to generate and plot a curve for a given clip.
    
    @clip_dir: directory of the clip, containing 100 frames in order
    @metric: metric to use to generate the curve
    @dry_run: if True, will generate a dummy curve
    '''
    ylim = {
        'psnr': (0, 50),
        'ssim': (0, 1),
    }
    xlim = {
        'psnr': (0, 100),
        'ssim': (0, 100),
    }

    def __init__(self,
                 clip_dir: str,
                 metric: function = Metrics.psnr,
                 dry_run: bool = False):
        assert os.path.exists(
            clip_dir), f'Clip directory {clip_dir} does not exist'
        assert os.path.isdir(
            clip_dir), f'Clip directory {clip_dir} is not a directory'
        assert IO.contain_frames(
            clip_dir), f'Clip directory {clip_dir} does not contain frames'

        self.clip_dir = clip_dir
        self.metric = metric
        os.makedirs(f'./res/{self.metric.__name__}', exist_ok=True)
        if dry_run:
            print(f'\tDry run for clip {self.clip_dir}')
            self.curve = [30 for i in range(99)]
        else:
            self.curve = self.gen_curve()

    def gen_curve(self):
        '''
        Generates the metric curve for the clip.
        
        Returns:
        a list of metric values for the curve
        '''
        frames = IO.get_frames(self.clip_dir)
        return list(Metrics.apply_pairwise(frames, self.metric))

    def plot(self, save: bool = False, save_name: str = "curve"):
        plt.plot(list(range(len(self.curve))), self.curve)
        plt.xlim(self.xlim.get(self.metric.__name__, None))
        plt.ylim(self.ylim.get(self.metric.__name__, (0, None)))
        if save:
            plt.savefig(f'./res/{self.metric.__name__}/{save_name}.png')
        else:
            plt.title(
                f'{self.metric.__name__.upper()} curve for clip {self.clip_dir}'
            )
            plt.show()
        plt.clf()


def test_dry_run():
    curve = Curve('./output/train/000/', dry_run=True)
    curve.plot()


def automation(dir: str,
               metric: function,
               clips_per_subdir=math.inf,
               depth: int = 1,
               dry_run: bool = False,
               save: bool = True) -> None:
    '''
    Automates the process of generating and plotting curves for 
    specified number of clips in one folder.
    
    Args:
    @metric: metric used to generate the curve
    @clips_per_subdir: number of clips to plot in each subdirectory
    @depth: depth of subdirectories to go into
    @dry_run: if True, will generate a dummy curve
    @save: if True, will save the curves as images
    '''
    if depth > 1:
        for subdir in IO.get_subdirs(dir):
            if dry_run:
                print(f'Subdir {subdir}')
            automation(subdir, metric, clips_per_subdir, depth - 1, dry_run,
                       save)
    else:
        clips = IO.get_subdirs(dir)
        for i, clip in enumerate(clips):
            if i >= clips_per_subdir:
                break

            curve = Curve(clip, metric=metric, dry_run=dry_run)
            curve_name = os.path.dirname(clip).replace(os.sep, '_')
            curve.plot(save=save, save_name=f'{curve_name}')


if __name__ == '__main__':
    test_dry_run()
    configs = {
        'dir': './output/',
        'metric': Metrics.psnr,
        'clips_per_subdir': 1,
        'depth': 2,
        'dry_run': False,
        'save': False,
    }

    automation(**configs)
