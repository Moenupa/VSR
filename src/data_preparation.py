import cv2
import glob
import os
import urllib.request
from pytube import YouTube
import logging
import json
import random

from config import Config

config = Config(stdout=False, dry_run=False)


def translate_video_id(fake_id: str) -> str:
    """Get the YouTube-format video ID from 4-character fake-id."""
    with urllib.request.urlopen(
            f'http://data.yt8m.org/2/j/i/{fake_id[:2]}/{fake_id}.js') as url:
        if url.getcode() != 200:
            raise Exception(f'translating {fake_id}: {url.getcode()} != 200')
        t = eval(url.read().decode('utf-8')[1:-1])
        if t[0] != fake_id:
            raise Exception(f'translating {fake_id}: {t[0]} != {fake_id}')
        return t[1]


def get_yt_url(youtube_id: str) -> str:
    return f'http://youtube.com/watch?v={youtube_id}'


def download_video(vid: str, download_path: str = './out/ytb/') -> bool:
    os.makedirs(download_path, exist_ok=True)
    try:
        url = get_yt_url(vid)
    except Exception as e:
        logging.error(f"[{url}] fails: {e}")
        return False
    all_streams = YouTube(url).streams.filter(mime_type="video/mp4",
                                              res="720p",
                                              only_video=True)
    if not all_streams:
        logging.warning(f"[{url}] fails: no option for 720p")
        return False

    stream = all_streams.last()
    logging.info(f"[{url}] started: {stream.filesize / 1024:.2f}MB")
    if not config.dry_run:
        stream.download(download_path, f'{vid}.mp4')
    logging.info(f"[{url}] downloaded {stream}")
    return True


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


def read_from_json(path: str = './res/tags.json') -> dict:
    file = open(path)
    d = json.load(file)
    file.close()
    return d


def download_batch(fake_ids: dict, batch_size: int = 10) -> tuple:
    batch = random.sample(list(fake_ids.items()), batch_size)
    logging.info(f"batch size={batch_size:03d} {batch}")
    downloaded, failed = {}, []
    for fake_id, labels in batch:
        vid = translate_video_id(fake_id)
        if download_video(vid):
            downloaded[vid] = labels
        else:
            failed.append(fake_id)
    return downloaded, failed


if __name__ == '__main__':
    fake_ids = read_from_json()
    # downloaded, failed = download_batch(fake_ids, 3)

    config.save_fp(lambda fp, obj: json.dump(obj, fp),
                   'downloaded.json',
                   obj=fake_ids)
