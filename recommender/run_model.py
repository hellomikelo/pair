import os
from itertools import product
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense

from style_stack import StyleStack
from run_style_similarity_search import *
from utils import plot_results, save_image
from configs import *


# TODO: add hypothesis and pytest testing
# TODO: package and move to tests dir

def main():
    # initialize model
    model = VGG16(weights='imagenet', include_top=False)
    
    if os.path.exists(FEATURE_LIBRARY_PATH):
        # library exists, load model
        print(f'==> Feature library exists. Loading...')
        stack = StyleStack.load(FEATURE_LIBRARY_PATH);
    else: 
        # library doesn't exist, build library
        print(f'==> Feature library does not exist. Building one now...')
        stack = StyleStack.build(
            image_dir=IMAGE_LIBRARY_PATH, 
            model=model, 
            layer_range=LAYER_NAMES)
        if SAVE_LIB:
            print(f'==> Saving embedding library')
            stack.save(FEATURE_LIBRARY_PATH)
        else: 
            print(f'==> Not saving embedding library')
    
    print(f'==> Run query on new image')
    results = stack.query(QUERY_IMAGE_PATH, None, N_RESULTS, write_output=False);    
    
    print(f'==> Query finished. Saving result image')
    save_image([QUERY_IMAGE_PATH] + results['results_files'], OUTPUT_IMAGE_PATH)

if __name__ == '__main__':
    main()
