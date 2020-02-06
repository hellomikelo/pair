import os
from itertools import product
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import SGD

from style_stack import StyleStack
from utils import plot_results, save_image, fbeta
from configs import *

def main():    
    if os.path.exists(FEATURE_LIBRARY_PATH):
        # library exists, load model
        print(f'==> Feature library exists. Loading...')
        stack = StyleStack.load(FEATURE_LIBRARY_PATH);
    else: 
        print(f'==> Feature library does not exist. Building one now...')
        # initialize pre-trained model
        if MODEL_TYPE is 'vgg16':
            print(f'==> Using VGG16')
            model = VGG16(weights='imagenet', include_top=False)
        elif MODEL_TYPE is 'transfer_learn':
            print(f'==> Using transfer learning model')
            model = load_model(MODEL_PATH, compile=False)
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
            image_dir=IMAGE_LIBRARY_PATH, 
            model=model, 
            layer_range=LAYER_NAMES)

        # save results
        if SAVE_LIB:
            print(f'==> Saving embedding library')
            stack.save(FEATURE_LIBRARY_PATH)
        else: 
            print(f'==> Not saving embedding library')
    
    print(f'==> Run query on new image')
    results = stack.query(QUERY_IMAGE_PATH, 
        embedding_weights=None, 
        n_results=N_RESULTS, 
        write_output=False);    

    truth = get_truth(TRUTH_FPATH)

    # TODO: hit rate at k
    eval = stack.hits_at_k(truth)
    
    print(f'==> Query finished. Saving result image')
    save_image([QUERY_IMAGE_PATH] + results['results_files'], OUTPUT_IMAGE_PATH)

if __name__ == '__main__':
    main()
