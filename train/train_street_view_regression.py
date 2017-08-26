from __future__ import print_function

import os
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error_regression, RotNetDataGenerator
from data.street_view import get_filenames as get_street_view_filenames


data_path = os.path.join('data', 'street_view')
train_filenames, test_filenames = get_street_view_filenames(data_path)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_street_view_resnet50_regression'

# input image shape
input_shape = (224, 224, 3)

# load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(1, activation='sigmoid', name='fc1')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# model compilation
model.compile(loss=angle_error_regression,
              optimizer='adam')

# training parameters
batch_size = 64
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
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        one_hot=False,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        one_hot=False,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, early_stopping, tensorboard],
    nb_worker=10,
    pickle_safe=True,
    verbose=1
)
