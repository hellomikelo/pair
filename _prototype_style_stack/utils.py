import datetime as dt
import json
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from configs import *
from PIL import Image


def load_image(path, target_size):
    # TODO: compare to vgg19.preprocess input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def save_image(results_paths, output_dir='../output/'):
    print(f'Result image saved to {output_dir}')
    combined = Image.new("RGB", (IMAGE_WIDTH*len(results_paths), IMAGE_HEIGHT))
    x_offset = 0
    for image in map(Image.open, results_paths):
        combined.paste(image, (x_offset, 0))
        x_offset += IMAGE_WIDTH
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
    print(f'keeping {len(image_paths)} image_paths to analyze')
    return image_paths


def get_concatenated_images(image_paths, image_indices=None, thumb_height=300):
    if image_indices is not None: 
        image_paths = [image_paths[i] for i in image_indices]
    thumbs = []
    for path in image_paths:
        img = image.load_img(path)
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def plot_results(results):
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


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join

    out_dir = '../output'
    paths = [join(out_dir, f) for f in listdir(out_dir) if isfile(join(out_dir, f))]
    for path in paths:
        plot_results(path)
