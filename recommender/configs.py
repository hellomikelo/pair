"""
Parameters to be loaded into the main model. 
"""
import time

# user inputs (CHANGE HERE!) 
# ==========================================================
# QUERY_IMAGE_PATH 		= '../data/bed/002.500.46.jpg'
# QUERY_IMAGE_PATH 		= '../data/bed/002.392.33.jpg'
# QUERY_IMAGE_PATH 		= '../data/bed/690.272.95.jpg'
QUERY_IMAGE_PATH 		= '../data/objects/902.782.77.jpg'
LAYER_NAMES 			= ['block1_conv1', 'block1_conv2']
LIB_TYPE 				= 'chair'
IMAGE_LIBRARY_PATH 		= '../data/' + LIB_TYPE + '/'
FEATURE_LIBRARY_PATH 	= '../output/indexes/feat_lib_' + LIB_TYPE + '/'
# OUTPUT_IMAGE_PATH 		= '../output/out-' + time.strftime('%Y%m%d-%H%M') + '.png'
OUTPUT_IMAGE_PATH		= f'../output/{LIB_TYPE}-out-{LAYER_NAMES[0]}-{LAYER_NAMES[1]}.png'
N_RESULTS 				= 4
SAVE_LIB 				= False
# ==========================================================

# ignore these
IMAGE_SIZE = 500
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE

# VGG16 layer names for content and style extraction
CONTENT_LAYERS = ["block2_conv2"]
STYLE_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]