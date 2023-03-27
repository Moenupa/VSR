import os
import cv2
import logging
import glob
import random
from multiprocessing import Pool

from src.config import Config

_720P = (1280, 720)
_360P = (640, 360)
_180P = (320, 180)


def video2frames(video_path: str,
                 out_base: str,
                 start_idx: int = -1,
                 n_frames: int = 100,
                 dry_run: bool = False,
                 gt_size: tuple = _720P) -> str:
    # get video id and format into output path
    # make sure lq and gt folders exists
    vid = os.path.splitext(os.path.basename(video_path))[0]
    lq_path = os.path.join(out_base, 'lq', vid)
    gt_path = os.path.join(out_base, 'gt', vid)
    os.makedirs(lq_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # checking dimensions and read video metadata
    if frame_size != gt_size:
        raise IOError(f'[{vid}] skipped, improper dimension: {frame_size}')
    if max_frames < n_frames:
        raise IOError(f'[{vid}] skipped, too short: {max_frames} < {n_frames}')

    count = 0
    if start_idx < 0:
        # try get n_frames from the middle, or from the start
        start_idx = random.randint(0, max_frames - n_frames)

    if not dry_run:
        # reading frames and save them into lq and gt folders
        # store frames [start_idx, ); len = n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        _, gt = cap.read()
        while count < n_frames:
            lq = cv2.resize(gt,
                            tuple(s // 4 for s in gt_size),
                            interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{os.path.join(lq_path, f'{count:06d}.png')}", lq)
            cv2.imwrite(f"{os.path.join(gt_path, f'{count:06d}.png')}", gt)
            _, gt = cap.read()
            count += 1
    cap.release()
    return f"[{vid}] -> frames [{start_idx},{start_idx + count}] from [0,{max_frames}]"


def clean(root: str):
    for path, _, _ in list(os.walk(root))[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def convert(clips: list, out: str, config: Config, n_threads: int = 32):
    with Pool(n_threads) as pool:
        for path in clips:
            pool.apply_async(
                video2frames,
                kwds={'video_path': path, 'out_base': out, 'dry_run': config.dry_run},
                callback=lambda msg: Config.log(config.stdout, logging.INFO, msg, config.log_dir),
                error_callback=lambda e: Config.log(config.stdout, logging.WARN, e, config.log_dir)
            )
        pool.close()
        pool.join()

    clean(f'{out}/lq')
    clean(f'{out}/gt')


if __name__ == '__main__':
    config = Config(stdout=False, dry_run=False)
    # convert(glob.glob(f"data/download/*.mp4"), 'data/STM', config)
    videos = glob.glob(f"data/YT8M/*.mp4")
    convert(videos, 'data/STM5k', config)
