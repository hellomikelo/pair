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

    if os.path.exists(cfg['FEATURE_LIBRARY_PATH']):
        # library exists, load model
        print(f'==> Feature library exists. Loading...')
        stack = StyleStack.load(cfg['FEATURE_LIBRARY_PATH'])
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

    # manually selected query images for checking HR@n
    query_imgs = [
        '202.493.30.jpg',
        '002.110.88.jpg',
        '090.066.63.jpg',
        '102.604.60.jpg',
        '003.015.26.jpg',
        '002.460.78.jpg',
        '102.191.78.jpg',
        '602.460.80.jpg',
        '900.954.28.jpg',
        '791.278.07.jpg',
        '702.068.04.jpg',
        '901.011.13.jpg',
        '991.278.11.jpg',
        '102.335.32.jpg',
        '500.395.52.jpg',
        '102.051.81.jpg',
        '701.032.50.jpg',
        '902.396.67.jpg',
        '490.904.81.jpg',
        '502.954.72.jpg'
    ]

    grd_truth = get_grd_truth(cfg['ITEM2ROOM'])
    hit_idx = []
    for i, query_img in enumerate(query_imgs):
        # print(f'==> Run query on {query_img}')
        results = stack.query(
            image_path='./data/chair/' + query_img, 
            lib_path=cfg['IMAGE_LIBRARY_PATH'],
            embedding_weights=None, 
            n_results=cfg['N_RESULTS'], 
            write_output=False);        
        # get hit rate at k
        hit_idx.append(get_hits_at_k(results, grd_truth))
        print(f'{query_img} \tHR@n: {hit_idx[i]} \tlen(results_files_all): {len(results["results_files_all"])}')

    # print output
    hra5 = sum([i <= 5 for i in hit_idx if i is not None])
    hra10 = sum([i <= 10 for i in hit_idx if i is not None])
    hra20 = sum([i <= 20 for i in hit_idx if i is not None])
    print(f'HR@5: {hra5} ({hra5/len(hit_idx)}%)')
    print(f'HR@10: {hra10} ({hra10/len(hit_idx)}%)')
    print(f'HR@20: {hra20} ({hra20/len(hit_idx)}%)')

if __name__ == '__main__':
    main()
