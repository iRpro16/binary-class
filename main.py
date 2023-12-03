import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam
import os
import cv2
import imghdr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#Avoid OOM errors
gpus = tf.config.experimental.list_logical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#Data, pizza = 1 & not pizza = 0
data_dir = 'binary-class/PizzaNotImages/pizza_not_pizza'
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#Scale data
data_scaled = data.map(lambda x,y: (x/255, y))
scaled_iterator = data_scaled.as_numpy_iterator()
batch_scaled = scaled_iterator.next()

#Split data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

train = data_scaled.take(train_size)
cv = data_scaled.skip(train_size).take(val_size)
test = data_scaled.skip(train_size+val_size).take(test_size)

