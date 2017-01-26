from __future__ import print_function

import os
import numpy as np
import argparse
from skimage.io import imsave
from skimage.transform import rotate

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import RotNetDataGenerator, crop_largest_rectangle, angle_error


def process_images(model, input_path, output_path,
                   batch_size=64, crop=True):
    extensions = ['.jpg', '.jpg', '.bmp', '.png']

    output_is_image = False
    if os.path.isfile(input_path):
        image_paths = [input_path]
        if os.path.splitext(output_path)[1] in extensions:
            output_is_image = True
            output_filename = output_path
            output_path = os.path.dirname(output_filename)
    else:
        image_paths = [os.path.join(input_path, f)
                       for f in os.listdir(input_path)
                       if os.path.splitext(f)[1] in extensions]
        if os.path.splitext(output_path)[1] in extensions:
            print('Output must be a directory!')

    predictions = model.predict_generator(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        ),
        val_samples=len(image_paths)
    )

    predicted_angles = np.argmax(predictions, axis=1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path, predicted_angle in zip(image_paths, predicted_angles):
        image = img_to_array(load_img(path))
        image = rotate(image, -predicted_angle, resize=False, preserve_range=True)
        if crop:
            image = crop_largest_rectangle(image, -predicted_angle)
        if not output_is_image:
            output_filename = os.path.join(output_path, os.path.basename(path))
        imsave(output_filename, image.astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model')
    parser.add_argument('input_path', help='Path to image or directory')
    parser.add_argument('-o', '--output_path', help='Output directory')
    parser.add_argument('-b', '--batch_size', help='Batch size for running the network')
    parser.add_argument('-c', '--crop', dest='crop', default=False, action='store_true',
                        help='Crop out black borders after rotating')
    args = parser.parse_args()

    print('Loading model...')
    model_location = load_model(args.model, custom_objects={'angle_error': angle_error})
    output_path = args.output_path if args.output_path else args.input_path

    print('Processsing input image(s)...')
    process_images(model_location, args.input_path, output_path,
                   args.batch_size, args.crop)
