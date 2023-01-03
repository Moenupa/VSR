import cv2
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np

def get_path(root_dir):
    return sorted(glob.glob(root_dir))
def psnr(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    psnr = cv2.PSNR(img1, img2)
    return psnr
def ssim(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    ssim = structural_similarity(img1, img2, multichannel=True)
    return ssim
def get_curve(paths, func = psnr):
    curve = []
    for i in range(len(paths)-1):
        point = func(paths[i], paths[i+1])
        curve.append(point)
    return curve
def fix_path(path):
    if path[-1] == '/':
        return path + '*'
    elif path[-1] == '*':
        return path
    else:
        return path + '/*'

if __name__ == '__main__':
    applied_metric = psnr
    output_dir = 'output/*/'
    # output_dir = 'data/pubg/*/'
    
    root_dirs = get_path(output_dir)
    for root_dir in root_dirs:
        clips = get_path(fix_path(root_dir))
        for clip in clips:
            paths = get_path(fix_path(clip))
            curve = get_curve(paths, func=applied_metric)
            print(f'Processing clip: {clip}, avg score: {np.mean(curve)}')
            # plot the curve
            plt.plot(list(range(len(curve))), curve)
            plt.ylim(bottom=0)
            if 'ssim' in applied_metric.__name__:
                plt.ylim((0, 1))
            plt.savefig(f'./res/{applied_metric.__name__}/{clip.replace("/", "_")}.png')
            plt.clf()