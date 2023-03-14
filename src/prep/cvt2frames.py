import os
import cv2

from src.config import Config

config = Config()

def video2frames(video_path: str,
                 lq_base: str,
                 gt_base: str,
                 start_idx: int = -1,
                 n_frames: int = 100,
                 lq_size: tuple = (320, 180)):
    # get video name
    file = os.path.basename(video_path)
    filename = os.path.splitext(file)[0]

    # make sure lq and gt folders exists
    lq_path = os.path.join(lq_base, filename)
    gt_path = os.path.join(gt_base, filename)
    os.makedirs(lq_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # checking dimensions and read video metadata
    if height != 720 or width != 1280:
        logging.warning(
            f'|{video_path}| skipped, improper dimension: {(width, height)}')
        return
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, max_frames)
    count = 0

    if not config.dry_run:
        # reading frames and save them into lq and gt folders
        if start_idx < 0:
            # try get n_frames from the middle, or from the start
            start_idx = max(0, (max_frames - n_frames) // 2)

        # store frames [start_idx, ); len = n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ret, gt = cap.read()
        while ret and count < n_frames:
            lq = cv2.resize(gt, lq_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{os.path.join(lq_path, f'{count:06d}.png')}", lq)
            cv2.imwrite(f"{os.path.join(gt_path, f'{count:06d}.png')}", gt)
            ret, gt = cap.read()
            count += 1
    cap.release()
    logging.info(
        f"|{video_path}| to frames [{start_idx},{start_idx + count}] from [0, {max_frames}]"
    )