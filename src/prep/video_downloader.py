from functools import partial
from multiprocessing import Pool, freeze_support
import os
import urllib.request
from pytube import YouTube
import logging
import json
import random
from yt_dlp import YoutubeDL

from config import Config

config = Config(stdout=False, dry_run=False, exp_code='download4')


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


<<<<<<< HEAD
def download_batch_ytb_dl(_batch: dict, download_path: str, dry_run: bool = False) -> None:
    logging.info(f"batch size={len(_batch):03d} {_batch}")

    '''with Pool(pool_size) as p:
        downloaded_list = p.map(partial(download_thread, path=path, dry_run=dry_run), _batch.keys())
        downloaded_entries = {vid: _batch[fid] for (fid, vid) in downloaded_list}
        logging.info(f'batch downloaded {sum([1 for _ in downloaded_list if _ is not None])}/{len(_batch)}')
        logging.info(f'batch entries: {downloaded_entries}')'''

    ydl_opts = {
        "format": "mp4[height=720]",
        "outtmpl": f"{download_path}/%(id)s.%(ext)s"
    }
    vid_batch = list(map(translate_video_id, _batch.keys()))

    if dry_run:
        return
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(vid_batch)


def download_video(vid: str, download_path: str, dry_run: bool = False) -> None:
    os.makedirs(download_path, exist_ok=True)
    if os.path.exists(f'{download_path}/{vid}.mp4'):
        return

    url = get_yt_url(vid)
    all_streams = YouTube(url).streams.filter(mime_type="video/mp4",
                                              res="720p",
                                              only_video=True)
    if not all_streams:
        raise ConnectionRefusedError(f"[{url}] fails: no option for 720p")

    stream = all_streams.last()
    logging.info(f"[{url}] started: {stream.filesize / 1024 / 1024:.2f}MB")
    if not dry_run:
        stream.download(download_path, f'{vid}.mp4')
    logging.info(f"[{url}] downloaded {stream}")


def read_from_json(path: str = './res/sample_labels.json') -> dict:
    file = open(path)
    d = json.load(file)
    file.close()
    return d


def download_thread(fake_id: str, path: str, dry_run: bool):
    try:
        vid = translate_video_id(fake_id)
        download_video(vid, path, dry_run)
    except ConnectionRefusedError as e:
        logging.warning(e)
    except Exception as e:
        logging.error(f'download error: {e}')
    else:
        return fake_id, vid


def download_batch(_batch: dict, path: str, pool_size: int = 4, dry_run: bool = False) -> None:
    logging.info(f"batch size={len(_batch):03d} {_batch}")

    with Pool(pool_size) as p:
        downloaded_list = p.map(partial(download_thread, path=path, dry_run=dry_run), _batch.keys())
        downloaded_entries = {vid: _batch[fid] for (fid, vid) in downloaded_list}
        logging.info(f'batch downloaded {sum([1 for _ in downloaded_list if _ is not None])}/{len(_batch)}')
        logging.info(f'batch entries: {downloaded_entries}')


def get_batch(_d: dict, _size: int = 100):
    return {k: v for k, v in random.sample(_d.items(), _size)}


if __name__ == '__main__':
    fakeid_with_label = read_from_json('./res/labels_100.json')
    iterations = 20

    for i in range(iterations):
        try:
            logging.info(f'starting iteration {i + 1} ---------->')
            batch = get_batch(fakeid_with_label)
            download_batch_ytb_dl(batch, f'./out/youTB/download', dry_run=config.dry_run)
        except Exception as err:
            logging.error(f'unknown error: {err}')
        finally:
            pass
            # config.save_fp(lambda fp, obj: json.dump(obj, fp), 'downloaded.json', obj=all_downloads)
