from __future__ import print_function

import argparse
import logging
import os
import sys

import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from tensorpack import logger

import google.cloud.logging
from google.cloud import datastore
from google.cloud.logging.handlers import CloudLoggingHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


parser = argparse.ArgumentParser(description='RotNet')
parser.add_argument("--log_dir", type=str, help="Log dir", required=True)
parser.add_argument("--run_id", type=str, help="Run id", required=True)

args = vars(parser.parse_args())

training_artifacts_dir = os.getenv("TRAINING_ARTIFACTS")
if not training_artifacts_dir:
    raise Exception("The env variable TRAINING_ARTIFACTS was not set.")
log_dir = os.path.join(
    training_artifacts_dir,
    args['log_dir'],
    args['run_id'],
    'logs'
)


# we don't need the labels indicating the digit value, so we only load the images
(X_train, _), (X_test, _) = mnist.load_data()
X_valid = X_train[50000:]
X_train = X_train[:50000]

target_img_size = (3 * 28, 3 * 28)

model_name = 'rotnet_mnist'

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# number of classes
nb_classes = 360

nb_train_samples = X_train.shape[0]
img_rows, img_cols = target_img_size
img_channels = 1
input_shape = (img_rows, img_cols, img_channels)
nb_test_samples = X_test.shape[0]

print('Input shape:', input_shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

# model definition
input = Input(shape=(img_rows, img_cols, img_channels))
x = Convolution2D(nb_filters, kernel_size)(input)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Convolution2D(nb_filters, kernel_size)(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128)(x)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=input, outputs=x)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[angle_error])

# training parameters
batch_size = 128
nb_epoch = 200

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
# checkpointer = ModelCheckpoint(
#     filepath=os.path.join(output_folder, model_name + '.hdf5'),
#     save_best_only=True
# )
# early_stopping = EarlyStopping(patience=2)
tensorboard = TensorBoard(log_dir=log_dir, batch_size=batch_size)


client = google.cloud.logging.Client()
handler = CloudLoggingHandler(client, name=args['run_id'])
cloudLogger = logging.getLogger("tensorpack")
cloudLogger.addHandler(handler)
cloudLogger.setLevel(logging.INFO)
cloudLogger.info('Logging started.')
client.setup_logging(log_level=logging.INFO)
logger.set_logger_dir(log_dir, "d")


# training loop
model.fit_generator(
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        target_img_size=target_img_size,
        shuffle=True
    ),
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_valid,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        target_img_size=target_img_size
    ),
    validation_steps=nb_test_samples//batch_size,
    verbose=1,
    callbacks=[tensorboard]
)

# display_examples(
#     model,
#     X_valid,
#     target_img_size=target_img_size,
#     save_path='img.png')
