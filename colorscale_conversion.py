# path = r'/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gesture/asl_alphabet_train/A/A26.jpg'

# img = Image.open(path).convert('L')
# img.save('greyscale_A1.jpg')
# inv_img = ImageChops.invert(img)
# inv_img.save('inverted_grayscale_A26.jpg')


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
        # print(joined_path)
        
        files = os.listdir(joined_path)
        for index, file in enumerate(files):
            # print(os.path.join(joined_path,file))
            img = Image.open(os.path.join(joined_path,file)).convert('L')
            inv_img = ImageChops.invert(img)
            inv_img.save('inverted_grayscale_'+file)
            # imgname  = 'inverted_grayscale_'+file'
            
            src_path = '/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/code/'+'inverted_grayscale_'+file
            dest_path = '/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gestures/inverted_kaggle_set/'+folder            
            shutil.move(src_path, dest)
