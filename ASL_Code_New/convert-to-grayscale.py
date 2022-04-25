from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import tensorflow_io as tfio
from PIL import Image 

path = r'/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gesture/asl_alphabet_train/A'

files = os.listdir(path)

for index, file in enumerate(files):
    # Your best bet is to attempt to open im = Image.open() and validate by calling im.load() all of your images, and record the filenames that fail.
    joined_path = os.path.join(path, file)
        
    img = Image.open(joined_path)
    
    image = tf.image.decode_jpeg(tf.io.read_file(img))
    print(image.shape, image.dtype)

    grayscale = tfio.experimental.color.rgb_to_grayscale(image)
    
    print(grayscale.shape, grayscale.dtype)
    # use tf.squeeze to remove last channel for plt.imshow to display:
    plt.figure()
    plt.imshow(tf.squeeze(grayscale, axis=-1), cmap='gray')
    plt.axis('off')
    plt.show()
    
    break # testing purposes
