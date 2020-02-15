from pair.style_stack import StyleStack
from pair.util import plot_results, save_image, fbeta, load_config, get_hits_at_k, get_grd_truth

import os
from itertools import product
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

def main():    
    cfg = load_config('config.yml')
    print(f'==> Model type: \t{cfg["MODEL_TYPE"]}')
    print(f'==> Query image: \t{cfg["QUERY_IMAGE_PATH"]}') 
    print(f'==> FEAT_LIB: \t{cfg["IMAGE_LIBRARY_PATH"]}\n')

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    if os.path.exists(cfg['FEATURE_LIBRARY_PATH']):
        # library exists, load model
        print(f'==> Feature library exists. Loading...')
        stack = StyleStack.load(
            lib_path=cfg['FEATURE_LIBRARY_PATH'],
            layer_range=cfg['LAYER_NAMES'])
    else: 
        print(f'==> Feature library does not exist. Building one now...')
        # initialize pre-trained model
        if cfg['MODEL_TYPE'] == 'vgg16':
            print(f'==> Using VGG16')
            model = VGG16(weights='imagenet', include_top=False)
        elif cfg['MODEL_TYPE'] == 'transfer_learn':
            print(f'==> Using transfer learning model')
            model = load_model(cfg['MODEL_PATH'], compile=False)
            # pop the top 3 layers (Flatten, Dense, Dense)
            model._layers.pop()
            model._layers.pop()
            model._layers.pop()
            model = Model(model.input, model.layers[-1].output)
            model.compile(
                optimizer=SGD(lr=0.01, momentum=0.9), 
                loss='binary_crossentropy', 
                metrics=[fbeta])

        # build embedding library using model
        stack = StyleStack.build(
            image_dir=cfg['IMAGE_LIBRARY_PATH'],
            lib_path=cfg['FEATURE_LIBRARY_PATH'],
            model=model, 
            layer_range=cfg['LAYER_NAMES'])

        # save embedding library for future use
        if cfg['SAVE_LIB']:
            print(f'==> Saving embedding library')
            stack.save(cfg['FEATURE_LIBRARY_PATH'])
        else: 
            print(f'==> Not saving embedding library')

    print(f'==> Run query on new image')
    results = stack.query(
        image_path=cfg['QUERY_IMAGE_PATH'], 
        lib_path=cfg['IMAGE_LIBRARY_PATH'],
        embedding_weights=None, 
        n_results=cfg['N_RESULTS'], 
        write_output=False);        

    # get item-to-room df
    grd_truth = get_grd_truth(cfg['ITEM2ROOM'])

    # get hit rate at k
    hit_idx = get_hits_at_k(results, grd_truth)
    
    print(f'==> Query finished. \tHR@k: {hit_idx}. \tResult image saved at {cfg["OUTPUT_IMAGE_PATH"]}')
    save_image(
        results_paths=[cfg['QUERY_IMAGE_PATH']] + results['results_files'], 
        output_dir=cfg['OUTPUT_IMAGE_PATH'],
        im_width=cfg['IMAGE_WIDTH'],
        im_height=cfg['IMAGE_HEIGHT'])

if __name__ == '__main__':
    main()
