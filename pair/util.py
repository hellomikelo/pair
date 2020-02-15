import datetime as dt
import json
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import yaml
import pickle
import pandas as pd
import math

from PIL import Image, ImageFont, ImageDraw

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_image(path, target_size=None):
    # TODO: compare to vgg19.preprocess input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# def save_image(results_paths, output_dir, im_width, im_height):
#     print(f'==> Result image saved to {output_dir}')
#     combined = Image.new("RGB", (im_width*len(results_paths), im_height))
#     x_offset = 0
#         combined.paste(image.resize((im_width, im_height)), (x_offset, 0))
#         x_offset += im_width
#     combined.save(output_dir) 

def save_image(results_paths, output_dir, im_width, im_height):
    print(f'==> Result image saved to {output_dir}')
    combined = Image.new("RGB", (im_width*len(results_paths), im_height))
    x_offset = 0
    count = 0

    # TODO: make output directory if it doesn't exist
    for image in map(Image.open, results_paths):
        img = image.resize((im_width, im_height))
        draw = ImageDraw.Draw(img)
        draw.text((20,20), results_paths[count], (0,0,0))
        combined.paste(img, (x_offset, 0))
        x_offset += im_width
        count += 1
    combined.save(output_dir) 

def get_image_paths(images_dir, max_num_images=10000):
    image_extensions = ['.jpg', '.png', '.jpeg']
    # TODO: shorten with glob
    # TODO: change to iterator instead of max_num_images
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in
                   os.walk(images_dir) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]

    if max_num_images < len(image_paths):
        image_paths = [image_paths[i] for i in sorted(random.sample(
            range(len(image_paths)), max_num_images))]
    print(f'==> Keeping {len(image_paths)} image_paths to analyze')
    return image_paths

def get_grd_truth(filepath):
    """ get items to room dataframe """
    f = open(filepath, 'rb')
    # f = tf.gfile.Open(os.path.join(args.job_dir, 'text_data/item_to_room.p'), 'rb')
    data = pickle.load(f)
    data = {key: [value] for key, value in data.items()}
    # make list of all room names
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['file_name', 'room_name']
    # clean up file names
    df['room_name'] = df['room_name'].map(lambda x: [line.split('/')[-1] for line in x])
    df['file_name'] = df.file_name.apply(lambda x: x.split('/')[-1])
    return df

def get_hits_at_k(results, grd_truth):
    """ calculate hit ratio at k as metric for recommendation quality"""
    query_id = results['query_img'].split('/')[-1]
    recom_ids = [fpath.split('/')[-1] for fpath in results['results_files_all']]
    query_rooms = grd_truth[grd_truth['file_name'] == query_id]['room_name'].iloc[0]
    hit_idx = None
    for idx, recom_id in enumerate(recom_ids):
        recom_rooms = grd_truth[grd_truth['file_name'] == recom_id]['room_name'].iloc[0]
        if set(query_rooms) & set(recom_rooms): 
            hit_idx = idx
            break
    return hit_idx

def get_concatenated_images(image_paths, image_indices=None, thumb_height=300):
    if image_indices is not None: 
        image_paths = [image_paths[i] for i in image_indices]
    thumbs = []
    for path in image_paths:
        # img = image.load_img(path[1:]) # [1:] to access current folder directory
        img = image.load_img(path) # [1:] to access current folder directory
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def get_result_images(results, n_results=4):
    if isinstance(results, str):
        with open(results) as f:
            json_str = json.load(f)
            results = {str(k): v for k, v in json_str.items()}

    results_files = results['results_files']
    if isinstance(results_files, dict):
        results_files = list(results_files.values())
    query_img_path = results['query_img']

    query_img = image.load_img(query_img_path)
    results_img = get_concatenated_images(results_files[:n_results])
    return query_img, results_img

def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score

def plot_results(results, n_results):
    if isinstance(results, str):
        with open(results) as f:
            json_str = json.load(f)
            results = {str(k): v for k, v in json_str.items()}

    results_files = results['results_files']
    if isinstance(results_files, dict):
        results_files = list(results_files.values())
    model = results['model']
    similarity_weights = results['similarity_weights']
    lib_name = results['lib_name']
    n_images = results['n_images']
    query_img_path = results['query_img']

    query_img = image.load_img(query_img_path)
    results_img_1 = get_concatenated_images(results_files[:n_results])
    results_img_2 = get_concatenated_images(results_files[5:])

    plt.figure(figsize=(8, 8))
    plt.subplot2grid((3, 1), (0, 0))
    plt.imshow(query_img)
    plt.title(f'{query_img_path}')

    plt.subplot2grid((3, 1), (1, 0))
    plt.imshow(results_img_1)
    plt.title(f'{similarity_weights}')

    plt.subplot2grid((3, 1), (2, 0))
    plt.imshow(results_img_2)
    plt.title(f'{lib_name}: {n_images} images')
    plt.figtext(0, 0, f'')

    plt.show()


def pdf_results(results_list, out_filename='pdf', incl_timestamp=True):
    pdf_dir = f'../output/pdfs/'
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if incl_timestamp:
        timestamp = str(dt.datetime.now())
    else:
        timestamp = ''
    pdf_path = os.path.join(pdf_dir, f'output_{out_filename}-{timestamp}.pdf')
    pdf_pages = PdfPages(pdf_path)

    for results in results_list:
        if isinstance(results, str):
            with open(results) as f:
                json_str = json.load(f)
                results = {str(k): v for k, v in json_str.items()}

        results_files = results['results_files']
        if isinstance(results_files, dict):
            results_files = list(results_files.values())
        model = results['model']
        similarity_weights = results['similarity_weights']
        lib_name = results['lib_name']
        n_images = results['n_images']
        query_img_path = results['query_img']

        query_img = image.load_img(query_img_path)
        results_img_1 = get_concatenated_images(results_files[:5])
        results_img_2 = get_concatenated_images(results_files[5:])

        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        ax1.imshow(query_img)
        ax1.set_xlabel(f'{query_img_path}', wrap=True)

        ax2 = plt.subplot2grid((3, 1), (1, 0))
        ax2.imshow(results_img_1)
        ax2.set_xlabel(f'{similarity_weights}', fontsize=6, wrap=True)

        ax3 = plt.subplot2grid((3, 1), (2, 0))
        ax3.imshow(results_img_2)
        ax3.set_xlabel(f'{lib_name}: {n_images} images', wrap=True)

        #plt.figtext(0, 0, f'')

        plt.tight_layout()
        pdf_pages.savefig(fig)
    pdf_pages.close()

def build_montages(image_paths, image_shape, montage_shape):
    """
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------
    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = 255 * np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False

    count = 0
    # font = ImageFont.truetype('/Library/Fonts/Arial.ttf', size=16)

    for path in image_paths:
        # if type(img).__module__ != np.__name__:
        #     raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = image.load_img(path)    
        img = img.resize(image_shape)
        draw = ImageDraw.Draw(img)
        draw.text((20,20), str(count), (0,0,0))
        img = np.array(img)

        # img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = 255 * np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), 
                    dtype=np.uint8)
                start_new_img = True
        count +=1 

    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    image_montages = np.asarray(image_montages)
    return image_montages    


# if __name__ == '__main__':
#     from os import listdir
#     from os.path import isfile, join

#     out_dir = '../output'
#     paths = [join(out_dir, f) for f in listdir(out_dir) if isfile(join(out_dir, f))]
#     for path in paths:
#         plot_results(path)
