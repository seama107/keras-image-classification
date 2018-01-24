import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import dataset
import cv2
import os

from cnn import *

from sklearn.metrics import confusion_matrix
from datetime import timedelta

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import History




# Config and Hyperparams

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 32             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 256

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['Pool', 'NonPool']
num_classes = len(classes)

# Dropout value
dropout = .2

# batch size
batch_size = 32

# test split
test_size = .16

# Num epochs
num_epochs = 100

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

data_path = '../pool-dataset'

data = dataset.read_train_sets(data_path, img_size, classes, test_size=test_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.images)))

# Building a model
model = Sequential()
model.add(Conv2D(filters=num_filters1,
                 kernel_size=filter_size1,
                 input_shape=(img_size, img_size, num_channels),
                 padding='same',
                 activation='relu',
                 kernel_constraint=maxnorm(3)))
model.add(Dropout(dropout))
model.add(MaxPooling2D())
model.add(Conv2D(filters=num_filters2,
                 kernel_size=filter_size2,
                 padding='same',
                 activation='relu',
                 kernel_constraint=maxnorm(3)))
model.add(Dropout(dropout))
model.add(MaxPooling2D())
model.add(Conv2D(filters=num_filters3,
                 kernel_size=filter_size3,
                 padding='same',
                 activation='relu',
                 kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(fc_size,
                activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(fc_size,
                activation='relu'))
model.add(Dense(2, activation='softmax'))
optimizer = Adam(lr=.01, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

model_dir = create_model_dir(model)
print("Saved model data to:", os.path.abspath(model_dir))


history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
callbacks = create_callbacks(model_dir)
history = train_model(model, data.train.images, data.train.labels, data.test.images, data.test.labels,
                      batch_size, callbacks, num_epochs, history)

print("Saving training history")
history_filename = os.path.join(model_dir, 'model_history.npy')
np.save(history_filename, history)
