import numpy as np
import cv2
from sklearn.datasets import load_files
from keras.utils import np_utils


def read_folders(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']))
    return files, targets


def load_data(files, targets, size):
    # Read images, create target variables
    data = [cv2.resize(src=cv2.imread(img), dsize=size) for img in files]
    x, y = np.array(data), np.array(targets)
    return x, y


# load train, test, and validation datasets
train_files, train_targets = read_folders('dogImages/train')
valid_files, valid_targets = read_folders('dogImages/valid')
test_files, test_targets = read_folders('dogImages/test')

x_train, y_train = load_data(train_files, train_targets, (224, 224))
x_test, y_test = load_data(test_files, test_targets, (224, 224))
x_valid, y_valid = load_data(valid_files, valid_targets, (224, 224))

np.save("saved_data/x_train", x_train)
np.save("saved_data/y_train", y_train)
np.save("saved_data/x_test", x_test)
np.save("saved_data/y_test", y_test)
np.save("saved_data/x_valid", x_valid)
np.save("saved_data/y_valid", y_valid)

check = np.load("saved_data/x_train.npy")
np.array_equal(x_train, check)

# Save custom image data
files, targets = read_folders('custom_images\\images')

import random

# Shuffle dataset
c = list(zip(files, targets))
random.shuffle(c)
files, targets = zip(*c)
del c
files = np.array(files)
targets = np.array(targets)

# split into train/test/valid
test_size = int(len(files) * .1)
valid_size = int(len(files) * .1)
train_start = test_size + valid_size
x_train, y_train = load_data(files[train_start:], targets[train_start:], (224, 224))
x_test, y_test = load_data(files[valid_size:train_start], targets[valid_size:train_start], (224, 224))
x_valid, y_valid = load_data(files[:valid_size], targets[:valid_size], (224, 224))


np.save("custom_images/data/x_train", x_train)
np.save("custom_images/data/y_train", y_train)
np.save("custom_images/data/x_test", x_test)
np.save("custom_images/data/y_test", y_test)
np.save("custom_images/data/x_valid", x_valid)
np.save("custom_images/data/y_valid", y_valid)

check = np.load("custom_images/data/x_train.npy")
np.array_equal(x_train, check)

# Check classes
from glob import glob
path = "custom_images/images"
classes = [name[len("custom_images/images/"):-1] for name in glob("custom_images/images/*/")]


# Balance the classes
import os
from shutil import copyfile
from random import shuffle


# Rename the files, remove the .jpg
person_list = os.listdir("custom_images/images/person")
cat_list = os.listdir("custom_images/images/cat")
dog_list = os.listdir("custom_images/images/dog")
other_list = os.listdir("custom_images/images/other")

len(person_list)
len(cat_list)
len(dog_list)
len(other_list)

# Normalize to 1000 images for each class
# person_list good no action
# Copy all person_list into custom_images/balanced/person
for person in person_list:
    copyfile("custom_images/images/person/" + person, "custom_images/balanced/person/" + person)
# cat_list copy cat_list 3x
for i, cat in enumerate(cat_list * 3):
    copyfile("custom_images/images/cat/" + cat, "custom_images/balanced/cat/" + str(i) + cat)
# dog_list copy dog_list 3x
i = 0
for i, dog in enumerate(dog_list * 3):
    copyfile("custom_images/images/dog/" + dog, "custom_images/balanced/dog/" + str(i) + dog)
# other_list shuffle, select firt 1000
shuffle(other_list)
for other in other_list[:1000]:
    copyfile("custom_images/images/other/" + other, "custom_images/balanced/other/" + other)

# Save the balanced data

# TODO start here! just finished balancing data, need to split between test/train/validate then save the npy
# todo then will be ready to test for 1 iteration, see if it's working, then train in full


# Save custom image data
files, targets = read_folders('custom_images\\balanced')

import random

# Shuffle dataset
c = list(zip(files, targets))
random.shuffle(c)
files, targets = zip(*c)
del c
files = np.array(files)
targets = np.array(targets)

# split into train/test/valid
test_size = int(len(files) * .1)
valid_size = int(len(files) * .1)
train_start = test_size + valid_size
x_train, y_train = load_data(files[train_start:], targets[train_start:], (224, 224))
x_test, y_test = load_data(files[valid_size:train_start], targets[valid_size:train_start], (224, 224))
x_valid, y_valid = load_data(files[:valid_size], targets[:valid_size], (224, 224))


np.save("custom_images/balanced/x_train", x_train)
np.save("custom_images/balanced/y_train", y_train)
np.save("custom_images/balanced/x_test", x_test)
np.save("custom_images/balanced/y_test", y_test)
np.save("custom_images/balanced/x_valid", x_valid)
np.save("custom_images/balanced/y_valid", y_valid)

check = np.load("custom_images/balanced/x_train.npy")
np.array_equal(x_train, check)



