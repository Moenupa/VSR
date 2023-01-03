import cv2
import glob
import os

def list_paths(dir):
    '''List all paths in dir
    
    Args:
    dir: path to directory, preferably a regex pattern
    
    Returns:
    a list of strings of all paths
    '''
    paths = glob.glob(dir)
    return paths

def video2frames(video_path, frames_path, start_idx = 0, num_frames = 125):
    '''Convert video to frames and save them in frames_path
    
    Args:
    video_path: path to video
    frames_path: path to save frames
    
    Returns:
    None
    '''
    os.makedirs(frames_path, exist_ok=True)
    vcap = cv2.VideoCapture(video_path)
    vcap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    res, img = vcap.read()
    count = 0
    while res and count < num_frames:
        frame = cv2.resize(img, (320,180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{ os.path.join(frames_path, f'{count:08d}.png')}", frame)
        res, img = vcap.read()
        count += 1
    print(f'Converted {video_path} frames [{start_idx},{start_idx+num_frames}]')
    
if __name__ == '__main__':
    v_360p = sorted(list_paths("data/*_360p.mp4"))
    v_720p = sorted(list_paths("data/*_720p.mp4"))
    
    print(f'Found Videos --- \n\t360P: {v_360p}\n\t720P: {v_720p}')
    
    targets = [('wot', 0), ('wot', 7530)]
    targets += [('pubg', 70), ('pubg', 450)]
    targets += []
    
    for (target, start_id) in targets:
        print(f'Processing: {target} to data/{target}')
        video2frames(f'data/{target}_360p.mp4', f'data/train_sharp_bicubic/{target}/{start_id}', start_idx=start_id)
        # video2frames(f'data/{target}_720p.mp4', f'data/train_sharp/{target}/{start_id}', start_idx=start_id)
    