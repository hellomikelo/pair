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

# Main stuff
def main():
	"""
	Create the style transfer model
	Modified from G. Surma's code
	"""
	# Initialize data
	input_image = backend.variable(input_image_array)
	style_image = backend.variable(style_image_array)
	combination_image = backend.placeholder((1, IMAGE_HEIGHT, IMAGE_SIZE, 3))

	# Load pre-trained VGG16
	input_tensor = backend.concatenate([input_image, style_image, combination_image], axis=0)
	model = VGG16(input_tensor=input_tensor, include_top=False)
	layers = dict([(layer.name, layer.output) for layer in model.layers])
	loss = backend.variable(0.)

	# Define content layer and loss
	layer_features = layers[CONTENT_LAYER]
	content_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]
	loss = loss + CONTENT_WEIGHT * content_loss(content_image_features, combination_features)

	# Define style layers and loss
	for layer_name in STYLE_LAYERS:
	    layer_features = layers[layer_name]
	    style_features = layer_features[1, :, :, :]
	    combination_features = layer_features[2, :, :, :]
	    style_loss = compute_style_loss(style_features, combination_features)
	    loss = loss + (STYLE_WEIGHT / len(style_layers)) * style_loss
	loss = loss + TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)
	
	outputs = [loss]
	outputs += backend.gradients(loss, combination_image)

	# Initialize output image and evaluator
	x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.
	evaluator = Evaluator()

	# Minimize cost function using L-BFGS
	# https://en.wikipedia.org/wiki/Limited-memory_BFGS
	for i in range(ITERATIONS):
	    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
	    print("Iteration %d completed with l_bfgs_loss %d" % (i, loss))
	    
    # Clean up result image
	x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
	x = x[:, :, ::-1]
	x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
	x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
	x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
	x = np.clip(x, 0, 255).astype("uint8")
	output_image = Image.fromarray(x)
	output_image.save(OUTPUT_IMAGE_PATH)
	print("Style transfer complete! Output image saved as %s" % (OUTPUT_IMAGE_PATH))

"""
def main():
	# From A McKay
    # build parameters
    image_dir = '../data/test'
    model = VGG16(weights='imagenet', include_top=False)
    lib_name = 'raw'
    n_results = 10

    # build, save, re-load, query, and plot
    stack = StyleStack.build(image_dir, model)
    stack.save(lib_name)
    del stack
    stack = StyleStack.load(lib_name)
    query_image = '../data/raw/735971-close-pins__cpin.png'
    results = stack.query(query_image, None, n_results, write_output=True)
    plot_results(results)
"""

if __name__ == '__main__':
	main()

