# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG16, MobileNet
from keras import optimizers
# from sklearn.preprocessing import MinMaxScaler
# import random
# from glob import glob
# import sys
from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":

    x_train = np.load("saved_data/x_train.npy")
    y_train = np.load("saved_data/y_train.npy")

    x_test = np.load("saved_data/x_test.npy")
    y_test = np.load("saved_data/y_test.npy")

    x_valid = np.load("saved_data/x_valid.npy")
    y_valid = np.load("saved_data/y_valid.npy")

    # Load in ResNet50 model with imagenet weights
    # conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # conv = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    conv = MobileNet(weights='imagenet', include_top=True, input_shape=(224, 224, 3), pooling=None)

    # Freeze the layers except the last 4 layers
    for layer in conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(133, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    # Setup the data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # zoom_range=0.05,
        horizontal_flip=True,
        # vertical_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

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
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=5e-6), metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.vggfinetune.hdf5',
                                   verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # Train the model
    history = model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        steps_per_epoch=len(x_train) / train_batchsize,
        validation_steps=len(x_valid) / val_batchsize,
        callbacks=[checkpointer, early_stopping],
        verbose=1)

    # Load weights and save the model
    model.load_weights('saved_models/weights.best.vggfinetune.hdf5')
    model.save('vggfinetune.h5')
    # FInished 11 iterations

    # Test accuracy
    # TODO START HERE! Load saved model
    model = models.load_model('vggfinetune.h5')
    predictions = model.predict_generator(test_datagen.flow(x_test, shuffle=False))
    test_accuracy = 100 * np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test , axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # lr: 1e-4, 60+ val and test acc, changed shape to vgg orig and added drop
    # lr: 1e-5, vgg shape, no drop out: 57% val
    # lr: 1e-5, vgg shape, w/ 50/40 drop out: 68% test: model.save("vgg_68.h5")
    # lr: 1e-5, vgg shape, w/ 30/10 drop out: 63% test
    # lr: 5e-5, vgg shape, w/ 50/40 drop out: 66.7% test

    # Add additional layers

    # Add additional nodes




    # plot visual
    df = pd.DataFrame(history.history)

    plt.plot(history.history)
    plt.show()
    plt.savefig('neural_net_plot.png')
