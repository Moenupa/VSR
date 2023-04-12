import os.path
import shutil

import cv2
import numpy as np
from PIL import Image
import glob
from constants import DATA_ROOT
from multiprocessing import Pool

Image.MAX_IMAGE_PIXELS = 1e9


def worker(img_path: str, box: tuple, save_path: str):
    img = Image.open(img_path)
    img.crop(box).save(save_path)
    img.close()


def crop_grid(img_path: str, size: tuple = (10, 10), padding: int = 2):
    img = Image.open(img_path)
    name, ext = os.path.basename(img_path).split('.')
    os.makedirs(name, exist_ok=True)
    w, h = img.size
    w, h = (w - padding) // size[0], (h - padding) // size[1]
    img.close()
    # w = cropped_w + padding
    with Pool(16) as p:
        for i in range(size[0]):
            for j in range(size[1]):
                p.apply_async(
                    worker,
                    kwds={
                        'img_path': img_path,
                        'box': (
                            i * w + padding,
                            j * h + padding,
                            (i + 1) * w,
                            (j + 1) * h
                        ),
                        'save_path': f'{name}/{i + j * size[1]:08d}.{ext}'
                    }
                )
        p.close()
        p.join()


def format_edvr_output(edvr_dir = 'data/STM3k/test30/edvr'):
    clips = glob.glob(f'{edvr_dir}/*')
    for clip in clips:
        files = glob.glob(f'{clip}/*/*.png')
        for file in files:
            shutil.move(file, f'{clip}/')


if __name__ == '__main__':
    format_edvr_output()
    os.chdir(f'{DATA_ROOT}/realbasicvsr')
    src_pics = glob.glob('*.png')
    print(src_pics)

    for pic in src_pics:
        print(f'\rcropping {pic}...', end='')
        crop_grid(pic, (10, 10))
