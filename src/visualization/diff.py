import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import time

root_dirs = [
    "output/train",
    "output/test",
    "output/basicvsr",
    "output/edvr",
    # "output/mfqe2",
]
rows = ['GT', 'LQ', 'BasicVSR', 'EDVR']
cols = ['img', 'feature1', 'feature2', 'feature3']


def find_paths(cat, idx):
    return list(f'{root_dir}/{cat}/{idx}.png' for root_dir in root_dirs)


def image_reshape(path, width=1280, height=720, feature=()):
    img = Image.open(path).convert('RGB')
    img = img.resize((width, height))
    if feature:
        img = img.crop(feature)
        img = img.resize((height, height))
    img = np.asarray(img)
    return img


def show_diff(paths):
    imgs = list(map(image_reshape, paths))
    imgs += list(
        image_reshape(path, feature=(200, 400, 400, 600)) for path in paths)
    imgs += list(
        image_reshape(path, feature=(1000, 500, 1200, 700)) for path in paths)
    imgs += list(
        image_reshape(path, feature=(688, 400, 800, 600)) for path in paths)

    fig = plt.figure(figsize=(16., 14.))
    grid = ImageGrid(
        fig,
        111,
        direction="column",
        nrows_ncols=(len(rows), len(cols)),
        axes_pad=0.5,
    )

    for id, (ax, im) in enumerate(zip(grid, imgs)):
        # Iterating over the grid returns the Axes.
        if id < len(rows):
            pass
            # print(f'PSNR{peak_signal_noise_ratio()} SSIM{}')
        ax.set_title(f'{rows[id % len(rows)]}_{cols[id // len(rows)]}')
        ax.imshow(im)
        # print(f'{rows[id % len(rows)]}_{cols[id // len(rows)]}')

    fig.tight_layout()
    plt.savefig(f"./res/diff/{int(time.time())}.png")


def compare_metrics(paths):
    imgs = list(cv2.resize(cv2.imread(path), (1280, 720)) for path in paths)
    for i in range(1, len(imgs)):
        psnr = peak_signal_noise_ratio(imgs[0], imgs[i])
        ssim = structural_similarity(imgs[0], imgs[i], channel_axis=2)
        print(f'Metrics for {rows[i]} --- PSNR: {psnr}, SSIM: {ssim}')


if __name__ == "__main__":
    idx = 99
    paths = find_paths("000", f"{idx:08d}")
    show_diff(paths)
    compare_metrics(paths)
