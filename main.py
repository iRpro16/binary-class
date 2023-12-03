import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.losses import BinaryCrossentropy
from keras.regularizers import L2
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

#Model

model = Sequential([
    Conv2D(16, (3,3), 1, activation='relu', input_shape = (256, 256, 3), kernel_regularizer=L2(0.01)),
    MaxPooling2D(),
    Conv2D(32, (3,3), 1, activation='relu', kernel_regularizer=L2(0.01)),
    MaxPooling2D(),
    Conv2D(16, (3,3), 1, activation='relu', kernel_regularizer=L2(0.01)),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=L2(0.1)),
    Dense(1, activation='sigmoid')
])

model.compile('adam',
              loss = BinaryCrossentropy(),
              metrics = ['accuracy'])

hist = model.fit(train, epochs=10, validation_data=cv)

#Plot model performance
fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()