import datetime as dt
import json
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from PIL import Image

"""
Data ingestion codes modified from Greg Surma's Github repo: 
https://github.com/gsurma/style_transfer
"""


#! Basic data IO


def get_img(IMAGE_WIDTH, IMAGE_HEIGHT):
	# From G. Surma
	input_image = Image.open(BytesIO(requests.get(SAN_FRANCISCO_IMAGE_PATH).content))
	input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
	input_image.save(INPUT_IMAGE_PATH)
	return input_image

def load_image(path, IMAGE_WIDTH, IMAGE_HEIGHT):
    # From A. McKay
    img = image.load_img(path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x	

def save_image(input_image, path): 
	# TBA
	pass 

def get_image_paths(images_dir, max_num_images=10000):
    # From A. McKay
    image_extensions = ['.jpg', '.png', '.jpeg']
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in
                   os.walk(images_dir) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]

    if max_num_images < len(image_paths):
        image_paths = [image_paths[i] for i in sorted(random.sample(
            range(len(image_paths)), max_num_images))]
    print(f'keeping {len(image_paths)} image_paths to analyze')
    return image_paths

def get_concatenated_images(image_paths, thumb_height=300):
	# From A. McKay
    thumbs = []
    for path in image_paths:
        img = image.load_img(path)
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def get_concatenated_matrices(matrices, thumb_height=300):
	# From A. McKay
    concat_matrices = np.concatenate([np.asarray(t) for mat in matrices], axis=1)
    return concat_matrices


#! Similarity search builder and helper functions


def _build_image_embedder(self, layer_range=None):
	# From A. McKay
    layer_names = [layer.name for layer in self.model.layers]
    if layer_range:
        slice_start = layer_names.index([layer_range[0]])
        slice_end = layer_names.index([layer_range[1]]) + 1
        chosen_layer_names = layer_names[slice_start:slice_end]
        chosen_layers = [layer for layer in self.model.layers
                         if layer.name in chosen_layer_names]
    else:
        chosen_layer_names = layer_names[1:]
        chosen_layers = self.model.layers[1:]
    self.layer_names = chosen_layer_names
    embedding_layers = [layer.output for layer in chosen_layers]
    self.embedder = K.function([self.model.input], embedding_layers)

@staticmethod
def gram_vector(x):
	# Compute Gram vector
	# From A. McKay
    if np.ndim(x) == 4 and x.shape[0] == 1:
        x = x[0, :]
    elif np.ndim != 3:
        # TODO: make my own error
        raise ValueError()
    x = x.reshape(x.shape[-1], -1)
    gram_mat = np.dot(x, np.transpose(x))
    mask = np.triu_indices(len(gram_mat), 1)
    gram_mat[mask] = None
    gram_vec = gram_mat.flatten()
    gram_vec = gram_vec[~np.isnan(gram_vec)]
    return gram_vec

def _build_query_gram_dict(self, img_embeddings):
	# From A. McKay
    gram_dict = {}
    for layer, emb in img_embeddings.items():
        gram_vec = self.gram_vector(emb)
        gram_vec = np.expand_dims(gram_vec, axis=0)
        if self._pca_id:
            transformer = self._load_transformer(self._pca_id, layer)
            gram_vec = transformer.transform(gram_vec)
        gram_dict[layer] = gram_vec
    return gram_dict

def _embed_image(self, image_path):
	# From A. McKay
    if self.model.input_shape[1]:
        _, x = load_image(image_path, self.model.input_shape[1:3])
    else:
        _, x = load_image(image_path, target_size=(224, 224))

    image_embeddings = self.embedder([x, 1])
    return image_embeddings

def _gen_lib_embeddings(self, image_paths):
	# From A. McKay
    for path in image_paths:
        try:
            image_embeddings = self._embed_image(path)
            self.valid_paths.append(path)
            yield image_embeddings
        except Exception as e:
            # TODO: add logging
            print(f'Embedding error: {e.args}')
            self.invalid_paths.append(path)
            continue

def _build_index(self):
	# From A. McKay
    start = dt.datetime.now()
    in_memory = True
    part_num = 0
    self.d_dict = {}
    self.index_dict = {}
    self.vector_buffer = [[] for _ in range(len(self.layer_names))]
    for i, img_embeddings in enumerate(self._embedding_gen):

        for k, emb in enumerate(img_embeddings):
            layer = self.layer_names[k]
            gram_vec = self.gram_vector(emb)
            self.vector_buffer[k].append(gram_vec)

            if i == 0:
                if self.pca_dim:
                    d = self.pca_dim    # int(len(gram_vec) * self.pca_frac)
                else:
                    d = len(gram_vec)
                self.index_dict[layer] = faiss.IndexFlatL2(d)
                self.d_dict[layer] = d

        if i % self.vector_buffer_size == 0 and i > 0:
            self._index_vectors()
            print(f'images {i - self.vector_buffer_size} - {i} indexed')
        if i % self.index_buffer_size == 0 and i > 0:
            in_memory = False
            part_num = ceil(i / self.index_buffer_size)
            self._save_indexes(self.lib_name, part_num)

    if self.vector_buffer:
        self._index_vectors()
        if not in_memory:
            part_num += 1
            self._save_indexes(self.lib_name, part_num)
            
    end = dt.datetime.now()
    index_time = (end - start).microseconds / 1000
    print(f'index time: {index_time} ms')

def _index_vectors(self):
    """
    Helper method to move data from buffer to index when
    `vector_buffer_size` is reached

    From A. McKay
    """
    if self.pca_dim:
        self._pca_id = dt.datetime.now()

    for j, gram_list in enumerate(self.vector_buffer):
        layer = self.layer_names[j]
        gram_block = np.stack(gram_list)
        if self.pca_dim:
            n, d = gram_block.shape

            # if more features than observations, PCA will return n
            # components, so we change dimensionality to n
            if n < d and self.index_dict[layer].ntotal == 0:
                self.index_dict[layer] = faiss.IndexFlatL2(n)
                self.d_dict[layer] = n
            transformer = PCA(self.d_dict[layer])
            gram_block = transformer.fit_transform(gram_block)
            self._save_transformer(layer, transformer)

        self.index_dict[layer].add(np.ascontiguousarray(gram_block))
        self.vector_buffer = [[] for _ in range(len(self.vector_buffer))]

def _save_indexes(self, lib_name, part_num):
	# From A. McKay
    if self.vector_buffer:
        self._index_vectors()

    self.lib_name = lib_name
    output_dir = f'../data/indexes/{lib_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for layer_name, index in self.index_dict.items():
        filename = f'grams-{layer_name}-part_{part_num}.index'
        filepath = os.path.join(output_dir, filename)
        faiss.write_index(index, filepath)
        self.index_dict = {}

    # save metadata
    if part_num == 1:
        metadata_path = os.path.join(output_dir, 'meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

# def _save_transformer(self, layer_name, transformer):
# 	# From A. McKay
#     transformer_dir = '../output/transformers/'
#     if not os.path.exists(transformer_dir):
#         os.makedirs(transformer_dir)
#     filename = f'pca-{self._pca_id}-{layer_name}'
#     transformer_path = os.path.join(transformer_dir, filename)
#     # with open(transformer_path, 'wb') as f:
#     job.dump(transformer, transformer_path)

# def _load_transformer(self, pca_id, layer_name):
# 	# From A. McKay
#     transformer_dir = '../output/transformers/'
#     filename = f'pca-{pca_id}-{layer_name}'
#     transformer_path = os.path.join(transformer_dir, filename)
#     # with open(transformer_path, 'rb') as f:
#     transformer = job.load(transformer_path)
#     return transformer


#! Similarity search main functions


def query(self, image_path, embedding_weights=None, n_results=5,
              write_output=True):
    # Query similarity between input image and Gram dictionary?
    # From A. McKay
    self._check_inputs_query(image_path, embedding_weights, n_results,
                             write_output)
    if not embedding_weights:
        embedding_weights = {name: 1 for name in self.layer_names}

    q_emb_list = self._embed_image(image_path)
    q_emb_dict = {layer: q_emb_list[i]
                  for i, layer in enumerate(self.layer_names) if layer in embedding_weights}
    query_gram_dict = self._build_query_gram_dict(q_emb_dict)

    start = dt.datetime.now()
    proximal_indices = set()
    for layer_name, gram in query_gram_dict.items():
        _, closest_indices = self.index_dict[layer_name].search(gram, n_results)
        proximal_indices.update(closest_indices[0].tolist())

    dist_dict = {}
    for layer_name, gram in query_gram_dict.items():
        labels_iter_range = list(range(1, len(proximal_indices) + 1))
        labels = np.array([list(proximal_indices), labels_iter_range])
        distances = np.empty((1, len(proximal_indices)), dtype='float32')
        self.index_dict[layer_name].compute_distance_subset(
            1, faiss.swig_ptr(gram), len(proximal_indices),
            faiss.swig_ptr(distances), faiss.swig_ptr(labels))
        distances = distances.flatten()
        norm_distances = distances / max(distances)
        dist_dict[layer_name] = {idx: norm_distances[i] for i, idx in
                                 enumerate(proximal_indices)}

    print(dist_dict)

    weighted_dist_dict = {}
    for idx in proximal_indices:
        weighted_dist = sum(
            [embedding_weights[layer_name] * dist_dict[layer_name][idx] for layer_name in
             embedding_weights])

        weighted_dist_dict[idx] = weighted_dist

    print(weighted_dist_dict)

    indices = sorted(weighted_dist_dict, key=weighted_dist_dict.get)
    results_indices = indices[:n_results]

    end = dt.datetime.now()
    index_time = (end - start).microseconds / 1000
    print(f'query time: {index_time} ms')
    print(results_indices)
    results_files = [self.file_mapping[i] for i in results_indices]
    results = {
        'query_img': image_path,
        'results_files': results_files,
        'similarity_weights': embedding_weights,
        'model': self.model.name,
        'lib_name': self.lib_name,
        'n_images': len(self.file_mapping),
        'invalid_paths': self.invalid_paths,
    }
    if write_output:
        timestamp = str(dt.datetime.now())
        output_dir = f'../output/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'results-{timestamp}')
        with open(output_file, 'w') as f:
            json.dump(results, f)
    return results

def query_distance(self, query_img_path, ref_path_list, embedding_weights):
	# NOT SURE IF FUNC IS USED
	# From A. McKay
    q_emb_list = self._embed_image(query_img_path)
    q_emb_dict = {layer: q_emb_list[i]
                  for i, layer in enumerate(self.layer_names) if layer in embedding_weights}
    query_gram_dict = self._build_query_gram_dict(q_emb_dict)

    start = dt.datetime.now()
    dist_dict = {}
    rev_file_mapping = {v: k for k, v in self.file_mapping.items()}
    ref_indices = [rev_file_mapping[path] for path in ref_path_list]
    for layer_name, gram in query_gram_dict.items():
        labels_iter_range = list(range(1, len(ref_indices) + 1))
        labels = np.array([list(ref_indices), labels_iter_range])
        distances = np.empty((1, len(ref_indices)), dtype='float32')
        self.index_dict[layer_name].compute_distance_subset(
            1, faiss.swig_ptr(gram), len(ref_indices),
            faiss.swig_ptr(distances), faiss.swig_ptr(labels))
        distances = distances.flatten()
        dist_dict[layer_name] = {idx: distances[i] for i, idx in
                                 enumerate(ref_indices)}


#! Misc visualization helper functions


def pdf_results(results_list, out_filename='pdf', incl_timestamp=True):
    # From A. McKay
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

def visualize_results():
	# From G. Surma
	# Visualize result image
	combined results
	combined = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
	x_offset = 0
	for image in map(Image.open, [INPUT_IMAGE_PATH, STYLE_IMAGE_PATH, OUTPUT_IMAGE_PATH]):
	    combined.paste(image, (x_offset, 0))
	    x_offset += IMAGE_WIDTH
	combined.save(COMBINED_IMAGE_PATH)
	return combined  

def plot_results(results):
	# From A. McKay
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
    results_img_1 = get_concatenated_images(results_files[:3])
    results_img_2 = get_concatenated_images(results_files[3:])

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