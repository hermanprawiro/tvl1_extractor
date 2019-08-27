import os
import glob
import numpy as np
import cv2
import subprocess
from PIL import Image

from flow_utils import readFlow, writeFlow

input_root = "E:\Datasets\Avenue"
output_root = "E:\\Datasets\\Avenue_Flow_TVL1_jpg"

def to_img(flow, bound=15):
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255/float(2*bound))
    return flow.astype(np.uint8)

def main():
    iext = 'jpg'
    tvl1 = cv2.DualTVL1OpticalFlow_create()
    folderlists = [
        os.path.join(input_root, 'Train'),
        os.path.join(input_root, 'Test'),
    ]
    for folder in folderlists:
        framefolders = os.listdir(folder)
        for framefolder in framefolders:
            images = sorted(glob.glob(os.path.join(folder, framefolder, '*.' + iext)))

            for i in range(len(images) - 1):
                output_path = os.path.join(output_root, os.path.dirname(os.path.relpath(images[i], input_root)))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                print('Processing', os.path.join(output_path, '{:06d}'.format(i)))

                im1 = np.array(Image.open(images[i]).resize((256, 256), Image.BILINEAR).convert('L'))
                im2 = np.array(Image.open(images[i + 1]).resize((256, 256), Image.BILINEAR).convert('L'))
                flow = tvl1.calc(im1, im2, None)
                flow = to_img(flow)
                Image.fromarray(flow[:, :, 0]).save(os.path.join(output_path, '{:06d}_u.jpg'.format(i)))
                Image.fromarray(flow[:, :, 1]).save(os.path.join(output_path, '{:06d}_v.jpg'.format(i)))

            
if __name__ == "__main__":
    main()