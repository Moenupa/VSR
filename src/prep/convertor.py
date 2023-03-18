import os
import cv2
import logging
import glob

from src.config import Config

_720P = (1280, 720)
_360P = (640, 360)
_180P = (320, 180)


def video2frames(video_path: str,
                 out_base: str,
                 start_idx: int = -1,
                 n_frames: int = 100,
                 dry_run: bool = False,
                 gt_size: tuple = _720P):
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
        logging.warning(f'[{vid}] skipped, improper dimension: {frame_size}')
        return
    n_frames = min(n_frames, max_frames)
    count = 0
    if start_idx < 0:
        # try get n_frames from the middle, or from the start
        start_idx = max(0, (max_frames - n_frames) // 2)

    if not dry_run:
        # reading frames and save them into lq and gt folders

        # store frames [start_idx, ); len = n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ret, gt = cap.read()
        while ret and count < n_frames:
            lq = cv2.resize(gt,
                            tuple(s // 4 for s in gt_size),
                            interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{os.path.join(lq_path, f'{count:06d}.png')}", lq)
            cv2.imwrite(f"{os.path.join(gt_path, f'{count:06d}.png')}", gt)
            ret, gt = cap.read()
            count += 1
    cap.release()
    logging.info(
        f"[{vid}] -> frames [{start_idx},{start_idx + count}] from [0,{max_frames}]"
    )


if __name__ == '__main__':
    config = Config(stdout=True, dry_run=False)
    videos = glob.glob(f"out/download/*.mp4")
    for video in videos:
        video2frames(video, 'out/frames', dry_run=False)
