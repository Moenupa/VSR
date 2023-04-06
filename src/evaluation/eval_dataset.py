import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_constants import *


def eval_dataset(path: str):
    df: pd.DataFrame = pickle.load(open(path, "rb")).round(2)
    fps_counts = df['fps'].value_counts()
    colors = 0.9 - np.random.rand(len(fps_counts), 3, ) / 2
    fig = plt.figure('STM FPS count', figsize=(1920/300, 1080/300), dpi=300, layout='tight')
    plt.scatter(fps_counts.index, fps_counts.values, s=fps_counts.values, c=colors)
    plt.yscale('log')
    plt.ylim(5e-1, 1e4)
    plt.ylabel('STM Dataset Clip Count')
    plt.xlabel('Frame Rate (FPS)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    eval_dataset(f"{DATA_ROOT}df.pkl")

