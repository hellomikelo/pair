"""
Prototype for product collection recommender

Mike Lo (2020)

---------
Implementation based on Andrew Look's work at Plato Designs
and Austin McKay's Style Stack work
https://github.com/TheAustinator/style-stack
"""

from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.decomposition import PCA
from scipy.spatial import distance
from PIL import Image

from utils import get_image_paths, load_image, get_concatenated_images, plot_results, pdf_results
from configs import *

def main():
    # A. McKay
    # initialize model 
    image_paths1 = get_image_paths('../data/raw')
    # image_paths2 = get_image_paths('../data/raw2')
    model = VGG16(weights='imagenet', include_top=True)
    layer_name, style_feature_model = get_embedding_model(model, 1)

    # content_feature_model = get_embedding_model(model, CONTENT_LAYERS[0])

    # generate embeddings
    valid_image_paths, style_embeddings = generate_embeddings(style_feature_model, image_paths1)
    # valid_image_paths, content_embeddings = generate_embeddings(content_feature_model, image_paths1)
    
    query_image_idx = int(len(valid_image_paths) * random.random())

    style_embeddings

    print(f'style_embeddings: {len(style_embeddings)}')
    # print(f'content_embeddings: {len(content_embeddings)}')

    # generate features
    style_features = pca(style_embeddings, None) #yhl
    # content_features = pca(content_embeddings, None) #yhl

    closest_image_indices_style = get_closest_indices(query_image_idx, style_features)
    # closest_image_indices_content = get_closest_indices(query_image_idx, content_features)
    
    results_image_paths_style = [valid_image_paths[i] for i in closest_image_indices_style]
    # results_image_paths_content = [valid_image_paths[i] for i in closest_image_indices_content]
    
    save_image(results_image_paths_style)
    # save_image(results_image_paths_content)

    # get results
    # query_image = get_concatenated_images(valid_image_paths, [query_image_idx])
    # results_image = get_concatenated_images(valid_image_paths, closest_image_indices)

    # print(f'embeddings shape is {len(embeddings)}')
    # print(f'features shape is {features.shape}')
    # print(f'query_image_idx = {query_image_idx}')
    # print(f'valid_image_paths are {valid_image_paths}')
    # print(f'closest_image_indices = {closest_image_indices}')
    # print(f'result image size = {results_image.shape}')

    # pdf_results(valid_image_paths)
    # plot_results(query_image_idx, query_image, closest_image_indices, results_image)

def get_embedding_model(model, layer_idx_from_output):
    layer_idx = len(model.layers) - layer_idx_from_output - 1
    layer_name = model.layers[layer_idx].name
    embedding_model = Model(inputs=model.input,
                            outputs=model.get_layer(f'{layer_name}').output)
    return layer_name, embedding_model

def _get_embedding_model(model, layer_name):
    # A. McKay
    embedding_model = Model(inputs=model.input,
                            outputs=model.get_layer(f'{layer_name}').output)
    return embedding_model


def generate_embeddings(embedding_model, image_paths, log_failures=False):
    # A. McKay
    embeddings = []
    valid_image_paths = []
    invalid_image_paths = []
    for i, image_path in enumerate(image_paths):
        if i % 1000 == 0:
            print("analyzing image %d / %d" % (i, len(image_paths)))
        try:
            _, x = load_image(image_path, embedding_model.input_shape[1:3])
        except Exception as e:
            invalid_image_paths.append(image_path)
            continue
        else:
            emb = embedding_model.predict(x)[0]
            embeddings.append(emb)
            valid_image_paths.append(image_path)  # only keep ones that didnt cause errors

    # TODO: add logging for invalid_images
    print(f'finished extracting {len(embeddings)} embeddings '
          f'for {len(valid_image_paths)} images with {len(invalid_image_paths)} failures')
    return valid_image_paths, embeddings


def pca(embeddings, n_components):
    # A. McKay
    embeddings = np.array(embeddings)
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    pca_features = pca.transform(embeddings)
    return pca_features


def get_closest_indices(query_image_idx, features, num_results=5, query_from_library=True):
    # A. McKay
    distances = [distance.cosine(features[query_image_idx], feat) for feat in features]
    start_idx = 1 if query_from_library else 0
    indices_closest = sorted(range(len(distances)), key=lambda k: distances[k])[
                      start_idx:start_idx + num_results + 1]
    return indices_closest


if __name__ == '__main__':
    main()
