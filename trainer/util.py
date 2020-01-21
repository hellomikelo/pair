import datetime as dt
import json
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import random

"""
Data ingestion codes modified from Greg Surma's Github repo: 
https://github.com/gsurma/style_transfer
"""

def get_img(IMAGE_WIDTH, IMAGE_HEIGHT):
	# From G. Surma
	input_image = Image.open(BytesIO(requests.get(SAN_FRANCISCO_IMAGE_PATH).content))
	input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
	input_image.save(INPUT_IMAGE_PATH)
	return input_image

def load_image(path, IMAGE_WIDTH, IMAGE_HEIGHT):
    # From A. McKay
    img = image.load_img(path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x	

def save_image(input_image, path): 
	TBA
	pass 

def visualize_results():
	# From G. Surma
	# Visualize result image
	combined results
	combined = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
	x_offset = 0
	for image in map(Image.open, [INPUT_IMAGE_PATH, STYLE_IMAGE_PATH, OUTPUT_IMAGE_PATH]):
	    combined.paste(image, (x_offset, 0))
	    x_offset += IMAGE_WIDTH
	combined.save(COMBINED_IMAGE_PATH)
	return combined
 