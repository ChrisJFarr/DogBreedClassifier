from keras.applications.mobilenet import MobileNet
from keras import Sequential, layers
from keras.callbacks import ModelCheckpoint
# Determine if image contains a dog, cat, human, or none of those


# TODO Pull in data
# TODO train with gpu



# Using mobile net for efficiency
# Only include convolutional layers
mobnet_conv = MobileNet(include_top=True, weights='imagenet', input_shape=(224, 224, 3))

for layer in mobnet_conv.layers:
    print(layer, layer.trainable)

model = Sequential()
model.add(mobnet_conv)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(479, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Check pointer for storing best model
checkpointer = ModelCheckpoint(filepath='model.weights.mobilenet.hdf5', verbose=1,
                               save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
                 validation_data=(x_valid, y_valid), callbacks=[checkpointer],
                 verbose=2, shuffle=True)