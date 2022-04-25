from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import tensorflow_io as tfio
from PIL import Image

path = '/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gesture/asl_alphabet_train/A'

files = os.listdir(path)

for index, file in enumerate(files):
    img = Image.open(os.path.join(path, file)) # open file
        
    image = tf.image.decode_jpeg(tf.io.read_file(img)) # read as jpeg file
    print(image.shape, image.dtype) # validate

    grayscale = tfio.experimental.color.rgb_to_grayscale(image)
    
    print(grayscale.shape, grayscale.dtype)
    # use tf.squeeze to remove last channel for plt.imshow to display:
    plt.figure()
    plt.imshow(tf.squeeze(grayscale, axis=-1), cmap='gray')
    plt.axis('off')
    plt.show()
    
    break # testing purposes
