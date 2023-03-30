import os.path as osp
import pickle
from random import randint

import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

DATA_ROOT = 'data/STM/test/'
COLORS_RGB = plt.rcParams["axes.prop_cycle"].by_key()["color"]

GT = {
    'gt': lambda clip_id, frame_id: f'{DATA_ROOT}gt/{clip_id:03}/{frame_id:08d}.png',
}

LQ = {
    'lq': lambda clip_id, frame_id: f'{DATA_ROOT}lq/{clip_id:03}/{frame_id:08d}.png',
}


def model_paths(model_names: list[str], display_names: list[str] = None):
    if display_names is None:
        display_names = model_names
    return {
        k: lambda clip_id, frame_id: f'{DATA_ROOT}{v}/{clip_id:03}/{frame_id:08d}/{frame_id:08d}.png'
        for k, v in zip(display_names, model_names)
    }


def basic_paths(path_names: list[str], display_names: list[str] = None):
    if display_names is None:
        display_names = path_names
    return {
        k: lambda clip_id, frame_id: f'{DATA_ROOT}{v}/{clip_id:03}/{frame_id:08d}.png'
        for k, v in zip(display_names, path_names)
    }


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_unpack(path):
    data = list(next(iter(record.values())) for record in pickle_read(path))
    df = pd.DataFrame(data)

    test_size, metrics = df.shape
    print(f'test_size: {test_size}, metrics: {metrics}')

    # figure, axes = plt.subplots(metrics, 1, sharex='all')
    clip_id = randint(0, test_size // 100 - 1) * 100
    print(f'selected clip_id: {clip_id // 100}')

    '''for i in range(metrics):
        ylim = (0, 1) if df.columns[i] == 'SSIM' else (0, None)
        df.iloc[clip_id:clip_id + 100, i].plot(
            ax=axes[i], legend=True,
            xlim=(clip_id - 10, clip_id + 110),
            ylim=ylim,
            color=colors[i],
        )
        axes[i].legend(loc='lower right')
    plt.show()'''
    plot_curve(
        df.iloc[:, :],
    )


def plot_curve(*args, **kwargs):
    curves = {i: v for i, v in enumerate(args)}
    curves.update(kwargs)
    # first num of rows, then columns
    shape = max(v.shape[-1] for v in curves.values()), len(curves)
    print(f'{shape}')

    figure, axes = plt.subplots(*shape, sharex='col', sharey='row')
    axes = axes.reshape(shape)
    for id_curves, (name, curve) in enumerate(curves.items()):
        for j, (colname, coldata) in enumerate(curve.items()):
            ax = axes[j, id_curves]
            ax.plot(coldata, label=colname, color=f'c{j}')
            ax.legend(loc='lower right')
            ax.relim()
            if 'PSNR' in colname:
                ax.set_ylim((20, 80))
            elif 'SSIM' in colname:
                ax.set_ylim((0.7, 1))
            else:
                ax.set_ylim((0, None))

    plt.show()


def extract_features(path, features: list[tuple[int, int, int, int]], width: int = 1280, height: int = 720) -> list:
    if not osp.exists(path):
        return [None] * (len(features) + 1)

    img = Image.open(path).resize((width, height))
    draw = ImageDraw.Draw(img)
    for i, f in enumerate(features):
        draw.rectangle((f[0], f[1], f[2] - 1, f[3] - 1), outline=COLORS_RGB[i], width=3)
    return [img] + [img.crop(f).resize((height, height)) for f in features]


def compare(path_interpreter: dict, clip_id: int | str, frame_id: int,
            features: list[tuple[int, int, int, int]] = None):
    if features is None:
        features = []

    # creating a figure, that is a grid with
    n_cols = len(features) + 1
    n_rows = len(path_interpreter)
    figure, axes = plt.subplots(
        n_rows, n_cols,
        sharex='col', sharey='all',
        width_ratios=[1280] + [720] * len(features),
        constrained_layout=True
    )
    axes = axes.reshape((n_rows, n_cols))
    np.reshape()

    for r, (display_name, v) in enumerate(path_interpreter.items()):
        path = v(clip_id, frame_id)
        print(path)
        extracted = extract_features(path, features)
        for c in range(n_cols):
            ax = axes[r, c]
            if img := extracted[c]:
                ax.imshow(img)
            if c == 0:
                ax.set_ylabel(display_name)
            if r == 0:
                ax.set_title(f'feature_{c}' if c > 0 else 'original')
            ax.set_yticks([])
            ax.set_xticks([])
    plt.show()


if __name__ == '__main__':
    # _ = pickle_unpack('data/STM/test/pred.pkl')
    compare(
        path_interpreter={**GT, **LQ, **model_paths(['edvr'])},
        clip_id='yy4kkeuywJY', frame_id=50,
        features=[(200, 400, 400, 600), (400, 400, 600, 600), (1000, 400, 1200, 600)]
    )
