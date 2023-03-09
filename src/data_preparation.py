import cv2
import glob
import os
import urllib.request
from pytube import YouTube
import logging

from config import Config

def translate_video_id(id):
    """Get the youtube video ID from 4-character string."""
    with urllib.request.urlopen(f'http://data.yt8m.org/2/j/i/{id[:2]}/{id}.js') as url:
        if url.getcode() != 200:
            raise Exception(f'translating {id}: {url.getcode()} != 200')
        t = eval(url.read().decode('utf-8')[1:-1])
        if t[0] != id:
            raise Exception(f'translating {id}: {t[0]} != {id}')
        return t[1]

def download_video(vid: str, download_path: str = './out') -> None:
    obj = YouTube(f'http://youtube.com/watch?v={vid}')
    try:
        obj.streams.get_by_itag(396).download(download_path, f'{obj.video_id}.mp4')
    except:
        obj.streams.get_by_itag(134).download(download_path, f'{obj.video_id}.mp4')
    finally:
        logging.info(f"downloaded video {vid}")
        

def video2frames(video_path: str, lq_base: str, gt_base: str, start_idx: int = -1, n_frames: int = 100, lq_size: tuple = (320, 180)):
    # get video name
    file = os.path.basename(video_path)
    filename = os.path.splitext(file)[0]
    
    # make sure lq and gt folders exists
    lq_path = os.path.join(lq_base, filename)
    gt_path = os.path.join(gt_base, filename)
    os.makedirs(lq_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    
    # reading frames and save them into lq and gt folders
    cap = cv2.VideoCapture(video_path)
    MAX_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, MAX_FRAMES)
    if start_idx < 0:
        # try get n_frames from the middle, or else last n_frames frames
        start_idx = min(MAX_FRAMES - n_frames, (MAX_FRAMES - n_frames) // 2)
        start_idx = max(0, start_idx)
    
    # store frames [start_idx, ); len = n_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ret, gt = cap.read()
    count = 0
    while ret and count < n_frames:
        lq = cv2.resize(gt, lq_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{ os.path.join(lq_path, f'{count:06d}.png') }", lq)
        cv2.imwrite(f"{ os.path.join(gt_path, f'{count:06d}.png') }", gt)
        ret, gt = cap.read()
        count += 1
    logging.info(f"converted {video_path} to frames [{start_idx},{start_idx + count}] from [0, {MAX_FRAMES}]")
    cap.release()

if __name__ == '__main__':
    config = Config(stdout=True)
    # print(translate_video_id('ABCD'))
    video2frames("data/.clips/wot_720p.mp4", "data/lq", "data/gt")
    
    
    
    