
"""
Hyperparameters modified from Greg Surma's Github repo: 
https://github.com/gsurma/style_transfer
"""
# Inputs
HOME_DIR = ''
INPUT_IMAGE_PATH = HOME_DIR + "input.png"
STYLE_IMAGE_PATH = HOME_DIR + "style.png"
OUTPUT_IMAGE_PATH = HOME_DIR + "output.png"
COMBINED_IMAGE_PATH = HOME_DIR + "combined.png"

# Hyperparameters
ITERATIONS = 2
CHANNELS = 3
IMAGE_SIZE = 500
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

# VGG16 layer names for content and style extraction
CONTENT_LAYER = "block2_conv2"
STYLE_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]