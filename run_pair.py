from pair.style_stack import StyleStack
from pair.utils import plot_results, save_image, fbeta, load_config

import os
from itertools import product
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import SGD

def main():    
    cfg = load_config('config.yml')
    print(f'model type: {cfg["MODEL_TYPE"]}')
    
    print(f'{cfg["MODEL_TYPE"] is "transfer_learn"}')

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
    
    print(f'==> Run query on new image')
    results = stack.query(
        image_path=cfg['QUERY_IMAGE_PATH'], 
        lib_path=cfg['IMAGE_LIBRARY_PATH'],
        embedding_weights=None, 
        n_results=cfg['N_RESULTS'], 
        write_output=False);    

    # TODO
    # truth = get_truth(cfg['TRUTH_FPATH'])

    # TODO: hit rate at k
    # eval = stack.hits_at_k(truth)
    
    # print(f'==> Query finished. Saving result image')
    # save_image([cfg['QUERY_IMAGE_PATH']] + results['results_files'], cfg['OUTPUT_IMAGE_PATH'])
    print(f'==> FINISHED!')

if __name__ == '__main__':
    main()
