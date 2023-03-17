import logging
from yt_dlp import YoutubeDL


def download_batch_ytb_dl(_batch: dict,
                          download_path: str,
                          dry_run: bool = False) -> None:
    logging.info(f"batch size={len(_batch):03d} {_batch}")
    '''with Pool(pool_size) as p:
        downloaded_list = p.map(partial(download_thread, path=path, dry_run=dry_run), _batch.keys())
        downloaded_entries = {vid: _batch[fid] for (fid, vid) in downloaded_list}
        logging.info(f'batch downloaded {sum([1 for _ in downloaded_list if _ is not None])}/{len(_batch)}')
        logging.info(f'batch entries: {downloaded_entries}')'''

    ydl_opts = {"format": "mp4[height=720]", "outtmpl": "%(id)s.%(ext)s"}
    vid_batch = list(map(translate_video_id, _batch.keys()))

    if dry_run:
        return
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(vid_batch)


if __name__ == '__main__':
    download_batch_ytb_dl(
        {'https://www.youtube.com/watch?v=BaW_jenozKc': [0, 1]}, './')
