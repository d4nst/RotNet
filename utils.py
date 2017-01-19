import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize

from keras.preprocessing.image import Iterator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import keras.backend as K


def angle_difference(x, y):
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def angle_error_regression(y_true, y_pred):
    return K.mean(angle_difference(y_true * 360, y_pred * 360))


def binarize_images(x):
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def crop_largest_rectangle(image, angle):
    height, width, _ = image.shape
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            math.radians(angle)
        )
    )


def generate_rotated_image(image, angle, size=None, crop_center=False,
                           crop_largest_rect=False):
    height, width, channels = image.shape
    if crop_center:
        crop_size = width if width < height else height
        image = crop_around_center(image, crop_size, crop_size)

    image = rotate(image, angle, resize=False, preserve_range=True)

    if crop_largest_rect:
        image = crop_largest_rectangle(image, angle)

    if size:
        new_width, new_height = size
        image = resize(
            image,
            (new_width, new_height, channels),
            preserve_range=True
        )

    return image


class RotNetDataGenerator(Iterator):

    def __init__(self, input, input_shape=None, color_mode='rgb', batch_size=64,
                 one_hot=True, preprocess_func=None, rotate=True, crop_center=False,
                 crop_largest_rect=False, shuffle=False, seed=None):

        self.images = None
        self.filenames = None
        self.input_shape = input_shape
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocess_func = preprocess_func
        self.rotate = rotate
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.shuffle = shuffle
        self.dim_ordering = K.image_dim_ordering()

        if self.color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', self.color_mode,
                             '; expected "rgb" or "grayscale".')

        if isinstance(input, (np.ndarray)):
            self.images = input
            N = self.images.shape[0]
            if not self.input_shape:
                self.input_shape = self.images.shape[1:]
        else:
            self.filenames = input
            N = len(self.filenames)

        super(RotNetDataGenerator, self).__init__(N, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.input_shape, dtype='float32')
        batch_y = np.zeros(current_batch_size, dtype='float32')

        grayscale = self.color_mode == 'grayscale'
        for i, j in enumerate(index_array):
            if self.filenames is None:
                image = self.images[j]
            else:
                image = load_img(self.filenames[j], grayscale=grayscale)
                image = img_to_array(
                    image,
                    dim_ordering=self.dim_ordering
                )

            if self.rotate:
                rotation_angle = np.random.randint(360)
            else:
                rotation_angle = 0

            rotated_image = generate_rotated_image(
                image,
                rotation_angle,
                size=self.input_shape[:2],
                crop_center=self.crop_center,
                crop_largest_rect=self.crop_largest_rect
            )

            batch_x[i] = rotated_image
            batch_y[i] = rotation_angle

        if self.one_hot:
            batch_y = to_categorical(batch_y, 360)
        else:
            batch_y /= 360

        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        return batch_x, batch_y


def display_examples(model, input, num_images=5, size=None, crop_center=False,
                     crop_largest_rect=False, preprocess_func=None, save_path=None):

    if isinstance(input, (np.ndarray)):
        images = input
        N, h, w, _ = images.shape
        if not size:
            size = (h, w)
        indexes = np.random.choice(N, num_images)
        images = images[indexes, :, :, :]
    else:
        images = []
        filenames = input
        N = len(filenames)
        indexes = np.random.choice(N, num_images)
        for i in indexes:
            image = load_img(filenames[i])
            images.append(img_to_array(image))
        images = np.asarray(images)

    x = []
    y = []
    for image in images:
        rotation_angle = np.random.randint(360)
        rotated_image = generate_rotated_image(
            image,
            rotation_angle,
            size=size,
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect
        )
        x.append(rotated_image)
        y.append(rotation_angle)

    x = np.asarray(x)
    y = np.asarray(y)

    y = to_categorical(y, 360)

    x_rot = np.copy(x)

    if preprocess_func:
        x = preprocess_func(x)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(x), axis=1)

    plt.figure(figsize=(10.0, 2 * num_images))

    title_fontdict = {
        'fontsize': 14,
        'fontweight': 'bold'
    }

    fig_number = 0
    for rotated_image, true_angle, predicted_angle in zip(x_rot, y, y_pred):
        original_image = rotate(
            rotated_image,
            -true_angle,
            resize=False,
            preserve_range=True,
        )
        if crop_largest_rect:
            original_image = crop_largest_rectangle(original_image, -true_angle)

        corrected_image = rotate(
            rotated_image,
            -predicted_angle,
            resize=False,
            preserve_range=True,
        )
        if crop_largest_rect:
            corrected_image = crop_largest_rectangle(corrected_image, -predicted_angle)

        if x.shape[3] == 1:
            options = {'cmap': 'gray'}
        else:
            options = {}

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 1:
            plt.title('Original\n', fontdict=title_fontdict)
        plt.imshow(np.squeeze(original_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 2:
            plt.title('Rotated\n', fontdict=title_fontdict)
        ax.text(
            0.5, 1.03, 'Angle: {0}'.format(true_angle),
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=11
        )
        plt.imshow(np.squeeze(rotated_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        reconstructed_angle = angle_difference(predicted_angle, true_angle)
        if fig_number == 3:
            plt.title('Reconstructed\n', fontdict=title_fontdict)
        ax.text(
            0.5, 1.03, 'Angle: {0}'.format(reconstructed_angle),
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=11
        )
        plt.imshow(np.squeeze(corrected_image).astype('uint8'), **options)
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if save_path:
        plt.savefig(save_path)

