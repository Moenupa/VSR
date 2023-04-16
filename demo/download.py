from yt_dlp import YoutubeDL

_720P = {"format": "22[fps>10]"}
_360P = {"format": "18[fps>10]"}
_144P = {"format": "160[fps>10]"}


def get_yt_url(youtube_id: str) -> str:
    return f'http://youtube.com/watch?v={youtube_id}'


def download_batch(_batch: list,
                   download_path: str,
                   res: int = 720,
                   dry_run: bool = False) -> int:

    ydl_opts = {
        "outtmpl": f"{download_path}/%(id)s.%(ext)s",
        "ignoreerrors": "True",
        # 'listformats': "True",
    }
    if res == 360:
        ydl_opts.update(_360P)
    elif res == 144:
        ydl_opts.update(_144P)
    else:
        ydl_opts.update(_720P)
    print(ydl_opts)

    if dry_run:
        return 0
    with YoutubeDL(ydl_opts) as ydl:
        ret = ydl.download(list(map(get_yt_url, _batch)))
        return ret


if __name__ == '__main__':
    print(download_batch(['X9AozZLZnCU'], 'demo', res=144, dry_run=False))