from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2
import random


# TODO add data augmentation
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)
# datagen.fit("data")
# When training
# model.fit_generator(datagen.flow("xtrain", "ytrain", batch_size=32), steps_per_epoch=)
# https://github.com/udacity/aind2-cnn/blob/master/cifar10-augmentation/cifar10_augmentation.ipynb

# (Optional) TODO: Report the performance of another
# face detection algorithm on the LFW dataset
# Feel free to use as many code cells as needed.

# TODO Build CNN face detector to distinguish between dog and human faces
# TODO human and dog files need to be the same dimensions, look into resize to smallest
# TODO Load in dog datasets


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# Dataset sizes
train_size = 2000
valid_size = 300
test_size = 300

# load train, test, and validation datasets
dog_train_files, train_targets = load_dataset('dogImages/train')
dog_valid_files, valid_targets = load_dataset('dogImages/valid')
dog_test_files, test_targets = load_dataset('dogImages/test')

# Match sizes of dataset samples, reserve remaining for further validation
dog_train_files = dog_train_files[:train_size]
dog_valid_files = dog_valid_files[:valid_size]
dog_test_files = dog_test_files[:test_size]

# TODO Load in human datasets
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)
# Split human_files into train, test, valid splits that match dog files
human_train_files, human_files = human_files[:train_size], human_files[train_size:]
human_valid_files, human_files = human_files[:valid_size], human_files[valid_size:]
human_test_files, human_files = human_files[:test_size], human_files[test_size:]
reserved_human_files = human_files[:]
del human_files

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0.01, 0.99))


# Read in data and create arrays with labels
def combine_shuffle_size_scale(dogs, humans, size):
    # Read images, create target variables
    h_list = [cv2.imread(img) for img in humans]
    h_labels = [[.99, 0.01] for _ in range(len(humans))]
    d_list = [cv2.imread(img) for img in dogs]
    d_labels = [[0.01, .99] for _ in range(len(dogs))]
    # Combine and shuffle dataset
    x = h_list + d_list
    y = h_labels + d_labels
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    # Resize images and convert to arrays
    x = [cv2.resize(src=img, dsize=size) for img in x]
    # rescale [0,255] --> [0,1]
    # x = scaler.fit_transform(x)
    x = [(img + 5) / 265 for img in x]  # Closer to min max scale 0.01 to 0.99
    # Convert to arrays
    x, y = np.array(x), np.array(y)
    return x, y


x_train, y_train = combine_shuffle_size_scale(dog_train_files, human_train_files, (250, 250))
x_valid, y_valid = combine_shuffle_size_scale(dog_valid_files, human_valid_files, (250, 250))
x_test, y_test = combine_shuffle_size_scale(dog_test_files, human_test_files, (250, 250))

# Pickle datasets
import pickle

pickle.dump(x_train, open("x_train.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))

pickle.dump(x_valid, open("x_valid.pkl", "wb"))
pickle.dump(y_valid, open("y_valid.pkl", "wb"))

pickle.dump(x_test, open("x_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

# human_list = [cv2.imread(img) for img in human_files_short]
# human_labels = [[.99, 0.01] for _ in range(len(human_list))]
# dog_list = [cv2.imread(img) for img in dog_files_short]
# dog_labels = [[0.01, .99] for _ in range(len(dog_list))]
#
# # Combine and shuffle dataset
# x_train = human_list + dog_list
# y_train = human_labels + dog_labels
# combined = list(zip(x_train, y_train))
# random.shuffle(combined)
# x_train, y_train = zip(*combined)
#
# # Resize images and convert to arrays
# x_train = [cv2.resize(src=img, dsize=(250, 250)) for img in x_train]
#
# # rescale [0,255] --> [0,1]
# x_train = [img/255 for img in x_train]
#
# # Convert to arrays
# x_train, y_train = np.array(x_train), np.array(y_train)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


# Use cpu, gpu out of memory
with tf.device("/cpu:0"):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(250, 250, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Check pointer for storing best model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                   save_best_only=True)

    hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
                     validation_data=(x_valid, y_valid), callbacks=[checkpointer],
                     verbose=2, shuffle=True)


score = model.evaluate(x_train, y_train, verbose=0)
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

# get predictions on the test set
y_hat = model.predict(x_test)
print("test")