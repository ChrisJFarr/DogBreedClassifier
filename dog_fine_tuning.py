# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG16, MobileNet, InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras import optimizers
# from sklearn.preprocessing import MinMaxScaler
# import random
# from glob import glob
# import sys
from matplotlib import pyplot as plt
import pandas as pd
# from keras.utils.generic_utils import CustomObjectScope


# Training steps:
# Start with pre-trained MobileNet
# Freeze all of the network and train the dense layers using adam
# Recompile the model and unfreeze the entire Mobile Net
# Load the best weights from the first training session
# Train the entire model using adam with a low learning rate


def build_model(freeze_pretrained=True):
    conv = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    if freeze_pretrained:
        for layer in conv.layers:
            layer.trainable = False

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(conv)

    # Add new layers
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1776, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1492, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(133, activation='softmax'))

    # Show summary for verification
    # Check the trainable status of the individual layers
    for layer in conv.layers:
        print(layer, layer.trainable)

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model


if __name__ == "__main__":

    x_train = np.load("saved_data/x_train.npy")
    y_train = np.load("saved_data/y_train.npy")

    x_test = np.load("saved_data/x_test.npy")
    y_test = np.load("saved_data/y_test.npy")

    x_valid = np.load("saved_data/x_valid.npy")
    y_valid = np.load("saved_data/y_valid.npy")

    # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
    #                         'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    #     model = load_model('weights.hdf5')
    # conv = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    model = build_model(True)
    # Setup the data generators

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Change the batchsize according to your system RAM
    train_batchsize = 32
    val_batchsize = 10
    image_size = 224
    train_dir = 'dogImages/train'
    valid_dir = 'dogImages/valid'
    test_dir = 'dogImages/test'

    train_generator = train_datagen.flow(x_train, y_train, batch_size=train_batchsize)
    validation_generator = validation_datagen.flow(x_valid, y_valid, batch_size=val_batchsize)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-4), metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.mobile_net_pre.hdf5',
                                   verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # Train the model
    history = model.fit_generator(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
        steps_per_epoch=len(x_train) // train_batchsize,
        validation_steps=len(x_valid) // val_batchsize,
        callbacks=[checkpointer, early_stopping],
        verbose=1)

    df = pd.DataFrame(history.history)
    # df.plot()
    df.plot()

    # Load weights and save the model
    model = build_model(False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-5), metrics=['accuracy'])
    model.load_weights('saved_models/weights.best.mobile_net_pre.hdf5')

    for layer in conv.layers:
        layer.trainable = True

    # Test accuracy
    # TODO START HERE! Load saved model
    # model = models.load_model('incresv2.h5')
    predictions = model.predict_generator(test_datagen.flow(x_test, shuffle=False))
    test_accuracy = 100 * np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(predictions)
    print('Test pre-accuracy: %.4f%%' % test_accuracy)

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.mobile_net.hdf5',
                                   verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.20,
        height_shift_range=0.20,
        zoom_range=0.20,
        shear_range=0.20,
        horizontal_flip=True,
        vertical_flip=False
    )

    train_generator = train_datagen.flow(x_train, y_train, batch_size=train_batchsize)

    history = model.fit_generator(
        train_generator,
        epochs=1000,
        validation_data=validation_generator,
        steps_per_epoch=(len(x_train) // train_batchsize) * 5,  # Why not 5x training data if aug.?
        validation_steps=len(x_valid) // val_batchsize,
        callbacks=[checkpointer, early_stopping],
        verbose=1)

    df = pd.DataFrame(history.history)
    # df.plot()
    df.plot()

    model.load_weights('saved_models/weights.best.mobile_net.hdf5')
    predictions = model.predict_generator(test_datagen.flow(x_test, shuffle=False))
    test_accuracy = 100 * np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

