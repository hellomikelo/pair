"""
Parameters to be loaded into the main model. 
"""
import time

# user inputs (CHANGE HERE!) 
# ==========================================================
QUERY_IMAGE_PATH 		= '../data/dining_table/902.224.07.jpg'
IMAGE_LIBRARY_PATH 		= '../data/chair/'
FEATURE_LIBRARY_PATH 	= '../output/indexes/feature_library/'
OUTPUT_IMAGE_PATH 		= '../output/result-' + time.strftime('%Y%m%d-%H%M') + '.png'
N_RESULTS 				= 5
# ==========================================================

# ignore these
IMAGE_SIZE = 500
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE

# VGG16 layer names for content and style extraction
CONTENT_LAYERS = ["block2_conv2"]
STYLE_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]