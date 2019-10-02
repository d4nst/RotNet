# RotNet

This repository contains the code necessary to train and test convolutional neural networks (CNNs) for predicting the rotation angle of an image to correct its orientation. There are scripts to train two models, one on [MNIST](http://yann.lecun.com/exdb/mnist/) and another one on the [Google Street View dataset](http://crcv.ucf.edu/data/GMCP_Geolocalization/). Since the data for this application is generated on-the-fly, you can also train using your own images in a similar way. A detailed explanation of the code and motivation for this project can be found in [my blog](https://d4nst.github.io/).

## Requirements
The code mainly relies on [Keras](https://keras.io/#installation) to train and test the CNN models, and [OpenCV](https://pypi.python.org/pypi/opencv-python) for image manipulation.

You can install all the required packages using pip: `pip install -r requirements.txt`

The recommended way to use Keras is with the TensorFlow backend. If you want to use it with the Theano backend you will need to make some minor modifications to the code to make it work.

## Train
Run either `python train/train_mnist.py` to train on MNIST or `python train/train_street_view.py` to train on the Google Street View dataset. Note that the first time you run the scripts will take longer since the datasets will be automatically downloaded. Also, you will need a decent GPU to train the *ResNet50* model that is used in `train_street_view.py`, otherwise it will take quite long to finish.

If you only want to test the models, you can download pre-trained versions [here](https://drive.google.com/open?id=0B9eNEi5uvOI1SjQ5M2tQY3ZMM1U).

Note also that the regression models (`train_mnist_regression.py` and `train_street_view_regression.py`) don't provide a good accuracy and are only included for illustration purposes.

## Test
You can evaluate the models and display examples using the provided Jupyter notebooks. Simply run `jupyter notebook` from the root directory and navigate to `test/test_mnist.ipynb` or `test/test_street_view.ipynb`.

Finally, you can use the `correct_rotation.py` script to correct the orientation of your own images. You can run it as follows:

`python correct_rotation.py <path_to_hdf5_model> <path_to_input_image_or_directory>`

You can also specify the following command line arguments:
- `-o, --output` to specify the output image or directory.
- `-b, --batch_size` to specify the batch size used to run the model.
- `-c, --crop` to crop out the black borders after rotating the images.

