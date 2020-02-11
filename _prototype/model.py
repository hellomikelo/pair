# Imports
import utils
import configs

import numpy as np
from PIL import Image
import requests
from io import BytesIO

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b

"""
Style transfer codes modified from Greg Surma's Github repo: 
https://github.com/gsurma/style_transfer

This is the main task for style transfer (to be modified into 
deep-collection recommender system)
"""

# Pre-define some functions for the model
def normalize_img(input_image): 
	# Data normalization and reshaping from RGB to BGR
	# From G. Surma's code
	input_image_array = np.asarray(input_image, dtype="float32")
	input_image_array = np.expand_dims(input_image_array, axis=0)
	input_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
	input_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
	input_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
	input_image_array = input_image_array[:, :, :, ::-1]
	return input_image_array

def gram_matrix(x):
	# Computes Gram matrix
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def total_variation_loss(x):
	# Computes TV
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

def compute_style_loss(style, combination):
	# Computes style loss
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))

def content_loss(content, combination):
	# Computes content loss
    return backend.sum(backend.square(combination - content))

def evaluate_loss_and_gradients(x):
	# Base function for computing loss and gradients
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:
	"""
	Evaluator class for computing loss and gradients.
	From G. Surma's code
	"""
    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients
