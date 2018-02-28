# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# Load in ResNet50 model with imagenet weights
res_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze the layers except the last 4 layers
for layer in res_conv.layers[:-10]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in res_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(res_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(133, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Setup the data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 50
val_batchsize = 10
image_size = 224
train_dir = 'dogImages/train'
valid_dir = 'dogImages/valid'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.resfinetune.hdf5',
                               verbose=1, save_best_only=True)
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=60,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    callbacks=[checkpointer],
    verbose=1)

# Load weights and save the model
model.load_weights('saved_models/weights.best.resnet_model.hdf5')
model.save('resnet.h5')
# FInished 11 iterations


