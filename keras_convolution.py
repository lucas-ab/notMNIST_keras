from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras.backend as K
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import TensorBoard, EarlyStopping


#-------Import dataset------
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale


#-------- Reformat the dataset to a new format -------
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#-----Generate the Convolutional model
batch_size = 16
patch_size = 3
depth = 16
num_hidden = 64

model = Sequential()
#---Convolutional layers---
model.add(Convolution2D(depth, patch_size, patch_size, border_mode='valid',
          input_shape=(image_size, image_size,num_channels),subsample = (1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(depth*2, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#---Fully connected layers---
model.add(Flatten())
model.add(Dense(1024, init = "he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512, init = "he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, init = "he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False),
            EarlyStopping(monitor='val_loss', patience=3, verbose=0)]

model.fit(train_dataset,train_labels,nb_epoch = 20, batch_size = batch_size,
          validation_data=(valid_dataset,valid_labels), callbacks = callbacks, shuffle=True)

score = model.evaluate(test_dataset, test_labels, batch_size=batch_size)

print ("\n", "Final test accuracy score: {}".format(score[1]))
