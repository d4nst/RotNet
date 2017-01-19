from __future__ import print_function

import os
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model

from utils import angle_error_regression, RotNetDataGenerator, binarize_images


# we don't need the labels indicating the digit value, so we only load the images
(X_train, _), (X_test, _) = mnist.load_data()

# add dimension to account for the channels (assuming tensorflow ordering)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

model_name = 'rotnet_mnist_regression'

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

nb_train_samples, img_rows, img_cols, img_channels = X_train.shape
input_shape = (img_rows, img_cols, img_channels)
nb_test_samples = X_test.shape[0]

print('Input shape:', input_shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

# model definition
input = Input(shape=(img_rows, img_cols, img_channels))
x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                  border_mode='valid',
                  input_shape=input_shape,
                  activation='relu')(input)
x = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                  border_mode='valid',
                  input_shape=input_shape,
                  activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input=input, output=x)

model.summary()


# model compilation
model.compile(loss=angle_error_regression,
              optimizer='adam')

# training parameters
batch_size = 128
nb_epoch = 50

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    save_best_only=True
)
early_stopping = EarlyStopping(patience=2)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        one_hot=False,
        preprocess_func=binarize_images,
        shuffle=True
    ),
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_test,
        one_hot=False,
        preprocess_func=binarize_images,
        batch_size=batch_size
    ),
    nb_val_samples=nb_test_samples,
    verbose=1,
    callbacks=[checkpointer, early_stopping, tensorboard]
)
