from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import tensorflow_io as tfio
import shutil
from PIL import Image, ImageChops

path = r'/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gesture/asl_alphabet_train'



folders = os.listdir(path)

for index, folder in enumerate(folders):
    print(folder)
    if folder != ".DS_Store":
        joined_path = os.path.join(path, folder)

        files = os.listdir(joined_path)
        for index, file in enumerate(files):
            img = Image.open(os.path.join(joined_path,file)).convert('L')
            # img.save('greyscale_'+file+'.jpg')
            inv_img = ImageChops.invert(img)
            inv_img.save('inverted_grayscale_'+file+'.jpg')
            imgname  = 'inverted_grayscale_'+file+'.jpg'
            
            source = '/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/code/'+imgname
            
            dest = path + '/' +folder # destination path 
            
            shutil.move(source, dest)

            break
