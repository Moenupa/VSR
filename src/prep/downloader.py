import os
from yt_dlp import YoutubeDL
import logging

from config import Config
from translator import YT8M_Translator


def get_yt_url(youtube_id: str) -> str:
    return f'http://youtube.com/watch?v={youtube_id}'


def download_batch_ytb_dl(_batch: list,
                          download_path: str,
                          dry_run: bool = False) -> None:
    logging.info(f"batch size={len(_batch):03d} {_batch}")

    ydl_opts = {
        "format": "22[fps>10]",
        "outtmpl": f"{download_path}/%(id)s.%(ext)s",
        "ignoreerrors": "True"
    }
    if dry_run:
        return
    with YoutubeDL(ydl_opts) as ydl:
        ret = ydl.download(list(map(get_yt_url, _batch)))
        logging.info(f'batch downloaded {ret}/{len(_batch)}')


if __name__ == '__main__':
    config = Config(stdout=True, dry_run=False)
    translator = YT8M_Translator()
    download_batch_ytb_dl(translator.get_all_vid()[:5], 'out/download')
