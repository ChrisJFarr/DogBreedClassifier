import numpy as np
import cv2
from sklearn.datasets import load_files
from keras.utils import np_utils


def read_folders(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 133)
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
