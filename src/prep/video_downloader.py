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
            raise ConnectionError(
                f'translating {fake_id}: {url.getcode()} != 200')
        t = eval(url.read().decode('utf-8')[1:-1])
        if t[0] != fake_id:
            raise ConnectionError(
                f'translating {fake_id}: {t[0]} != {fake_id}')
        return t[1]


def get_yt_url(youtube_id: str) -> str:
    return f'http://youtube.com/watch?v={youtube_id}'


def download_video(vid: str,
                   download_path: str = f'./out/ytb/{config.exp_code}'
                   ) -> bool:
    os.makedirs(download_path, exist_ok=True)

    url = get_yt_url(vid)
    all_streams = YouTube(url).streams.filter(mime_type="video/mp4",
                                              res="720p",
                                              only_video=True)
    if not all_streams:
        logging.warning(f"[{url}] fails: no option for 720p")
        return False

    stream = all_streams.last()
    logging.info(f"[{url}] started: {stream.filesize / 1024 / 1024:.2f}MB")
    if not config.dry_run:
        stream.download(download_path, f'{vid}.mp4')
    logging.info(f"[{url}] downloaded {stream}")
    return True


def read_from_json(path: str = './res/sample_labels.json') -> dict:
    file = open(path)
    d = json.load(file)
    file.close()
    return d


def download_batch(fake_ids: dict, batch_size: int = 10) -> tuple:
    batch = random.sample(list(fake_ids.items()), batch_size)
    logging.info(f"batch size={batch_size:03d} {batch}")
    downloaded, failed = {}, []
    for fake_id, labels in batch:
        try:
            vid = translate_video_id(fake_id)
            if download_video(vid):
                downloaded[vid] = labels
            else:
                failed.append(fake_id)
        except Exception as e:
            logging.error(f'download error: {e}')
            failed.append(fake_id)
    return downloaded, failed


if __name__ == '__main__':
    fake_ids = read_from_json('./res/labels_100.json')
    iterations = 100
    batch_size = 5
    all_downloads = {}

    for i in range(iterations):
        try:
            logging.info(f'iteration {i}, batch {batch_size}')
            chunk, _ = download_batch(fake_ids, batch_size)
            all_downloads = {**chunk, **all_downloads}
        except Exception as e:
            logging.error(f'unknown error: {e}')
        finally:
            config.save_fp(lambda fp, obj: json.dump(obj, fp),
                           'downloaded.json',
                           obj=all_downloads)
