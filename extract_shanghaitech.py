import os
import glob
import numpy as np
import cv2
import subprocess
from PIL import Image

input_root = "E:\\Datasets\\ShanghaiTech"
output_root = "E:\\Datasets\\ShanghaiTech_Flow_TVL1_S2"
visualization_root = "E:\\Datasets\\ShanghaiTech_Flow_TVL1_Vis"

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

            for i in range(len(images) - 2):
                output_path = os.path.join(output_root, os.path.dirname(os.path.relpath(images[i], input_root)))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                print('Processing', os.path.join(output_path, '{:06d}'.format(i)))

                im1 = np.array(Image.open(images[i]).resize((256, 256), Image.BILINEAR).convert('L'))
                im2 = np.array(Image.open(images[i + 2]).resize((256, 256), Image.BILINEAR).convert('L'))
                flow = tvl1.calc(im1, im2, None)
                flow = to_img(flow)
                Image.fromarray(flow[:, :, 0]).save(os.path.join(output_path, '{:06d}_u.jpg'.format(i)))
                Image.fromarray(flow[:, :, 1]).save(os.path.join(output_path, '{:06d}_v.jpg'.format(i)))

def generate_visualization():
    f2i_root = 'D:\\Codes\\utils\\flow2image'
    f2i_main = os.path.join(f2i_root, 'f2i.py')

    folderlists = [
        os.path.join(output_root, 'Train'),
        os.path.join(output_root, 'Test'),
    ]
    
    for folder in folderlists:
        framefolders = os.listdir(folder)
        for framefolder in framefolders:
            flow_path = os.path.join(folder, framefolder)
            visualization_path = os.path.join(visualization_root, os.path.relpath(flow_path, output_root))
            
            script_path = [
                'python', f2i_main, '-v', '-r', '25',
                os.path.join(flow_path, '*.flo'),
                '-o', visualization_path
            ]

            subprocess.run(script_path)
            
if __name__ == "__main__":
    main()
    # generate_visualization()