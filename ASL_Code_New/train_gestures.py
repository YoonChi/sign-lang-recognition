import pandas as pd
import numpy as np
import tensorflow as tf

batch_size = 32
img_height=256
img_width=256

train_ds = tf.keras.utils.image_dataset_from_directory(
    '/Users/anjalia/Desktop/AI_PROJECT-ASL/archive/asl_alphabet_train',
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    '/Users/anjalia/Desktop/AI_PROJECT-ASL/archive/asl_alphabet_train',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

class_names = train_ds.class_names
print("Class name: ", class_names)
print("Total classes: ", len(class_names))

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(len(class_names)):
        ax = plt.subplot(6,5,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    layers.Rescaling(1./255,input_shape=(256,256,3)),
    layers.Conv2D(16, 3, activation='relu'), #padding ≠ same, as output value size ≠ input volume size
    layers.MaxPooling2D(), #downsampled

    layers.Conv2D(32, 3, activation='relu', padding = 'same'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding = 'same'),
    layers.MaxPooling2D(),

    layers.Flatten(),

#model.add(Dropout(0.2))
    layers.Dense(128,activation ="relu"),
#model.add(Dropout(0.3))
    layers.Dense(29,activation ="softmax")
])

model.summary()

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, batch_size=32,validation_batch_size=32, validation_data=test_ds,epochs=3)

