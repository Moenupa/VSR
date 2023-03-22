import os
from yt_dlp import YoutubeDL
import logging

from config import Config
from translator import YT8M_Translator


def chunk(_l: list, _size: int) -> list:
    for i in range(0, len(_l), _size):
        yield _l[i:i + _size]


def get_yt_url(youtube_id: str) -> str:
    return f'http://youtube.com/watch?v={youtube_id}'


def download_batch(_batch: list,
                   download_path: str,
                   dry_run: bool = False) -> int:
    logging.info(f"batch size={len(_batch):03d} {_batch}")

    ydl_opts = {
        "format": "22[fps>10]",
        "outtmpl": f"{download_path}/%(id)s.%(ext)s",
        "ignoreerrors": "True",
    }
    if dry_run:
        return 0
    with YoutubeDL(ydl_opts) as ydl:
        ret = ydl.download(list(map(get_yt_url, _batch)))
        logging.info(f'batch downloaded {ret}/{len(_batch)}')
        return ret


def download_all(_video_ids: list, download_path: str, n: int = 1000, _batch_size=10,
                 dry_run: bool = False) -> int:
    s = 0
    while s < n:
        batch = chunk(_video_ids, _batch_size)
        s += download_batch(batch, download_path, dry_run)
    return s


if __name__ == '__main__':
    config = Config(stdout=False, dry_run=False)
    translator = YT8M_Translator()
    video_id = translator.get_all_vid()[3000:]
    download_all(video_id, 'data/YT8M')
