from .util import load_image, get_image_paths

from abc import ABC, abstractclassmethod, abstractmethod
import datetime as dt
import faiss
import glob
import joblib as job
import json
import tensorflow.keras.applications as apps
import tensorflow.keras.backend as K
from math import ceil
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
import random
import re
import yaml

import streamlit as st
import _thread
import tensorflow

class Stack(ABC):
    models = {
        'densenet121': apps.densenet.DenseNet121,
        'densenet169': apps.densenet.DenseNet169,
        'densenet201': apps.densenet.DenseNet201,
        'inceptionv3': apps.inception_v3.InceptionV3,
        'inceptionresnetv2': apps.inception_resnet_v2.InceptionResNetV2,
        'mobilenet': apps.mobilenet.MobileNet,
        'mobilenetv2': apps.mobilenet_v2.MobileNetV2,
        'nasnetlarge': apps.nasnet.NASNetLarge,
        'nasnetmobile': apps.nasnet.NASNetMobile,
        'resnet50': apps.resnet50.ResNet50,
        'vgg16': apps.vgg16.VGG16,
        'vgg19': apps.vgg19.VGG19,
        'xception': apps.xception.Xception,
    }

    def __init__(self, lib_path):
        self.valid_paths = []
        self.invalid_paths = []
        self.vector_buffer_size = None
        self.index_buffer_size = None
        self.pca_dim = None
        self.model = None
        self.layer_names = None
        self._file_mapping = None
        self._partitions = None
        self._transformer = None
        self._pca_id = None
        self.lib_path = lib_path

    @classmethod
    @abstractmethod
    def build(cls, image_dir, model, layer_range, pca_dim,
              vector_buffer_size, index_buffer_size, max_files):
        pass

    @classmethod
    @abstractmethod
    def load(cls, lib_path, layer_range=None, model=None):
        pass

    @abstractmethod
    def save(self, lib_path):
        pass

    @abstractmethod
    def query(self, image_path, embedding_weights=None, n_results=5,
              write_output=True):
        pass

    @property
    def metadata(self):
        return {
            'model': self.model.name,
            'layer_names': self.layer_names,
            'paritions': self.partitions,
            'pca': str(self._pca_id),
        }

    @property
    def partitions(self):
        if self._partitions is None:
            index_paths = glob.glob(os.path.join(self.lib_path, 'index-*.index'))
            if not index_paths:
                self._partitions = False
            else:
                sample_path = index_paths[0]
                if 'part' in sample_path:
                    self._partitions = False
                    part_nums = set()
                    for f in index_paths:
                        info = re.search(f'{input_dir}index-(.+?)-part(.+?)\.index', f)
                        part_num = info.group(2)
                        part_nums.add(part_num)
                    n_parts = max(part_nums)
                    self._partitions = n_parts
                else:
                    self._partitions = False
        return self._partitions

    @partitions.setter
    def partitions(self, value):
        self._partitions = value

    @property
    def file_mapping(self):
        if self._file_mapping:
            pass
        else:
            self._file_mapping = {i: f for i, f in enumerate(self.valid_paths)}
        return self._file_mapping

    @file_mapping.setter
    def file_mapping(self, value):
        self._file_mapping = value

class StyleStack(Stack):
    """
    This class is used to do style similarity search of a query image against
    a libaray of images. The search uses the l2 difference between the gram
    matrices of user-selected embeddings from a user specified convolutional
    network. The similarity search uses Facebook AI Research's faiss library
    to for speed and scalability, and the `StyleStack` class acts as a high level
    ORM to the index.

    This class is not instantiated directly, but constructed via either the
    `StyleStack.build` class method to build a new set of indexes from raw
    images, or the `StyleStack.load` constructor to load a set of indexes from
    disk. More images can always be added to an existing index using the `add`
    method on an instance of `StyleStack`. The `query` method is used to get the
    filepaths of the most similar images. The weighting of different layer
    embeddings in the similarity search can be adjusted at query time.

    Example:
        Building a StyleStack
        >>> image_dir = '../data/my_data'
        >>> model = apps.vgg16.VGG16(weights='imagenet', include_top=False)
        >>> layer_range = ('block1_conv1', 'block2_pool')
        >>> stack = StyleStack.build(image_dir, model, layer_range)
        Saving
        >>> lib_name = 'library_name'
        >>> stack.save(lib_name)
        Loading
        >>> stack = StyleStack.load(lib_name)
        Querying
        >>> # any embeddings not in embedding_weights will not be used
        >>> image_path = '../data/my_data/cat_painting.jpg'
        >>> embedding_weights = {
        >>>     'block1_conv1': 1,
        >>>     'block3_conv2': 0.5,
        >>>     'block3_pool': .25
        >>> }
        >>> n_results = 5
        >>> results = stack.query(image_path, n_results, embedding_weights,
        >>>                       write_output=True)
    """
    @classmethod
    def build(cls, image_dir, lib_path, model, layer_range=None, pca_dim=None,
              vector_buffer_size=2000, index_buffer_size=5000, max_files=2000):
        """
        Use this constructor when you do not have a preexisting gram matrix
        library built by another instance of `StyleStack`.

        This is the first of two constructors for `StyleStack`, which builds a
        set of faiss indexes, one for each embedding layer of the model. Each
        index is specific to the model and layer that were used to embed it, so
        a new `StyleStack` should be built if the `model` is changed or if
        any of the layers used to `query` are not in `layer_range` of the prior
        gram matrix library. However, if the layers used to query are a subset
        of `layer_range` and the `model` and images are the same, use `load`.

        Args:
            image_dir (str): directory containing images to be indexed

            model (tensorflow.keras.engine.training.Model): model from which to extract
                embeddings.

            layer_range (None, iterable[str]): A two item iterable containing the
                desired first and last layer names to be inclusively used as
                layer selection bounds. Alternatively, `None` uses all layers
                except for input. Note that the gram matrix computations do not
                work on dense layers, so if the model has dense layers, they
                should be excluded with this argument or loaded with
                `include_top=False` in tensorflow.keras.

            vector_buffer_size (int): number of embeddings to load into memory
                before indexing them. Reduce this if your system runs out of
                memory before printouts that some items have been indexed.

            index_buffer_size (int): number of files to index before forcing indexes
                to be saved to disk. If the number of image files is less than
                this, than the indexes will be held in memory. Otherwise, the
                partitioned indexes are automatically saved to disk when they
                reach the size of `index_buffer_size`. Reduce this when running
                out of memory subsequent to printouts that some items have been
                indexed.

        Returns:
            inst (cls): instance of `cls` built from the gram matrices of the
                embeddings generated by the layers of the `model` specified by
                the `layer_range`. Note that the underlying indexes are not
                saved unless the `inst.save` method is run.
        """
        inst = cls(lib_path)
        inst.vector_buffer_size = vector_buffer_size
        inst.index_buffer_size = index_buffer_size
        inst.pca_dim = pca_dim
        image_paths = get_image_paths(image_dir)
        random.shuffle(image_paths)
        image_paths = image_paths[:max_files]
        if isinstance(model, str):
            model_cls = cls.models[model]
            model = model_cls(weights='imagenet', include_top=False)
        inst.model = model
        inst._build_image_embedder(layer_range)
        inst._embedding_gen = inst._gen_lib_embeddings(image_paths)
        inst._build_index(lib_path)
        return inst

    # @st.cache(hash_funcs={_thread.LockType: type, _thread.RLock: type})

    @classmethod 
    @st.cache(hash_funcs={_thread.LockType: type, _thread.RLock: type}, allow_output_mutation=True)
    def load(cls, lib_path, layer_range=None, model=None):
        """

        Args:
            lib_name:

            layer_range:

            model: Model to embed query images. It must be the same as the model
                used to embed the reference library. If `None`, the model name
                will be gathered from the metadata, and it will be loaded from
                `tensorflow.keras.applications` with `weights='imagenet'` and
                `include_top=False`. If the model used to build the original
                `StyleStack` is not part of `tensorflow.keras.applications` or did not use
                imagenet weights, the model will not be generated correctly from
                metadata and must be passed in via this argument

        Returns:

        """
        inst = cls(lib_path)

        # invalid paths have already been filtered out
        inst.invalid_paths = None

        # load metadata
        with open(os.path.join(lib_path, 'meta.json')) as f:
            json_str = json.load(f)
            metadata = {str(k): v for k, v in json_str.items()}
        inst._pca_id = metadata['pca']

        # load model
        if model is None:
            model_str = metadata['model']
            model_cls = StyleStack.models[model_str]
            model = model_cls(weights='imagenet', include_top=False)
        inst.model = model

        # build embedder from model
        inst._build_image_embedder(layer_range)

        # load file mapping
        with open(os.path.join(lib_path, 'file_mapping.json')) as f:
            json_str = json.load(f)
            inst.file_mapping = {int(k): str(v) for k, v in json_str.items()}

        # load indexes and check for partitioning
        index_paths = glob.glob(os.path.join(lib_path, 'grams-*.index'))
        sample_path = index_paths[0]
        if 'part' in sample_path:
            inst.partitioned = True

        # load indexes into memory
        if not inst.partitions:
            inst.index_dict = {}
            for f in index_paths:
                index = faiss.read_index(f)
                layer_name = re.search(f'{lib_path}grams-(.+?)\.index', f).group(1)
                inst.index_dict.update({layer_name: index})

        # set up generator to load partitions
        else:
            # TODO finish this
            raise NotImplementedError(
                'Loading partitioned indexes not implemented yet')

        return inst

    def add(self, image_dir):
        pass

    def save(self, lib_path):
        if not os.path.exists(lib_path):
            os.makedirs(lib_path)
        for layer_name, index in self.index_dict.items():
            filename = f'grams-{layer_name}.index'
            filepath = os.path.join(lib_path, filename)
            faiss.write_index(index, filepath)

        mapping_path = os.path.join(lib_path, 'file_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.file_mapping, f)

        metadata_path = os.path.join(lib_path, 'meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def query(self, image_path, lib_path, embedding_weights=None, n_results=5,
              write_output=True):
        # TODO: refactor
        # TODO: create seperate query class, which has attributes like distances by layer, etc. This will be cleaner and allow sliders without re-running query
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

        weighted_dist_dict = {}
        for idx in proximal_indices:
            weighted_dist = sum(
                [embedding_weights[layer_name] * dist_dict[layer_name][idx] for layer_name in
                 embedding_weights])

            weighted_dist_dict[idx] = weighted_dist

        indices = sorted(weighted_dist_dict, key=weighted_dist_dict.get)
        results_indices = indices[:n_results]

        end = dt.datetime.now()
        index_time = (end - start).microseconds / 1000
        
        print(f'==> Query time: {index_time} ms')
        results_files = [self.file_mapping[i] for i in results_indices]
        
        results_files_all = [self.file_mapping[i] for i in indices]
        if results_files[0][:2] == '..':
            # TODO: debug
            # changes '../' to './'
            results_files = [fpath[1:] for fpath in results_files] 
            results_files_all = [fpath[1:] for fpath in results_files_all]
        self.results_files = results_files

        results = {
            'query_img': image_path,
            'results_files_all': results_files_all,
            'results_files': self.results_files,
            'similarity_weights': embedding_weights,
            'model': self.model.name,
            'lib_path': lib_path,
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

    def hits_at_k(self, truth):
        # hit rate at k as a metric for recommender system
        # truth:    

        # find all products that appear in same room as query product

        # check which recommended products also appear in product catalog
        pass



    def query_dist(self, query_img_path, ref_path_list, embedding_weights):
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

    @staticmethod
    def gram_vector(x):
        if np.ndim(x) == 4 and x.shape[0] == 1:
            x = x[0, :]
        elif np.ndim != 3:
            raise ValueError('Array dimension mismatch')
        x = x.reshape(x.shape[-1], -1)
        gram_mat = np.dot(x, np.transpose(x))
        mask = np.triu_indices(len(gram_mat), 1)
        gram_mat[mask] = None
        gram_vec = gram_mat.flatten()
        gram_vec = gram_vec[~np.isnan(gram_vec)]
        return gram_vec

    def _build_query_gram_dict(self, img_embeddings):
        gram_dict = {}
        for layer, emb in img_embeddings.items():
            gram_vec = self.gram_vector(emb)
            gram_vec = np.expand_dims(gram_vec, axis=0)
            # if self._pca_id: #yhl
            #     transformer = self._load_transformer(self._pca_id, layer)
            #     gram_vec = transformer.transform(gram_vec)
            gram_dict[layer] = gram_vec
        return gram_dict

    def _build_image_embedder(self, layer_range=None):
        print(f'==> Using layers ranging between {layer_range}')
        layer_names = [layer.name for layer in self.model.layers]
        if layer_range:
            slice_start = layer_names.index(layer_range[0])
            slice_end = layer_names.index(layer_range[1]) + 1
            chosen_layer_names = layer_names[slice_start:slice_end]
            chosen_layers = [layer for layer in self.model.layers
                             if layer.name in chosen_layer_names]
        else:
            chosen_layer_names = layer_names[1:]
            chosen_layers = self.model.layers[1:]
        self.layer_names = chosen_layer_names
        embedding_layers = [layer.output for layer in chosen_layers]
        self.embedder = K.function([self.model.input], embedding_layers)

    # TODO: output successful embeddings to make resumable
    def _gen_lib_embeddings(self, image_paths):
        for path in image_paths:
            try:
                image_embeddings = self._embed_image(path)
                self.valid_paths.append(path)
                yield image_embeddings

            except Exception as e:
                # TODO: add logging
                print(f'==> Embedding error: {e.args}')
                self.invalid_paths.append(path)
                continue

    def _embed_image(self, image_path):
        if self.model.input_shape[1]:
            _, x = load_image(image_path, self.model.input_shape[1:3])
        else:
            _, x = load_image(image_path, target_size=(224, 224))

        image_embeddings = self.embedder([x, 1])
        return image_embeddings

    # TODO: split into gen_gram_matrices and _build_index, then combine gen_gram_matrices with build_query_gram_dict
    def _build_index(self, lib_path):
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
                print(f'==> images {i - self.vector_buffer_size} - {i} indexed')

            if i % self.index_buffer_size == 0 and i > 0:
                in_memory = False
                part_num = ceil(i / self.index_buffer_size)
                self._save_indexes(lib_path, part_num)

        if self.vector_buffer:
            self._index_vectors()
            if not in_memory:
                part_num += 1
                self._save_indexes(lib_path, part_num)

        end = dt.datetime.now()
        index_time = (end - start).microseconds / 1000
        print(f'==> index time: {index_time} ms')

    def _index_vectors(self):
        """
        Helper method to move data from buffer to index when
        `vector_buffer_size` is reached
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

    def _save_indexes(self, lib_path, part_num):
        if self.vector_buffer:
            self._index_vectors()

        if not os.path.exists(lib_path):
            os.makedirs(lib_path)
        for layer_name, index in self.index_dict.items():
            filename = f'grams-{layer_name}-part_{part_num}.index'
            filepath = os.path.join(lib_path, filename)
            faiss.write_index(index, filepath)
            self.index_dict = {}

        # save metadata
        if part_num == 1:
            metadata_path = os.path.join(lib_path, 'meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)

    def _save_transformer(self, layer_name, transformer):
        transformer_dir = '../output/transformers/'
        if not os.path.exists(transformer_dir):
            os.makedirs(transformer_dir)
        filename = f'pca-{self._pca_id}-{layer_name}'
        transformer_path = os.path.join(transformer_dir, filename)
        # with open(transformer_path, 'wb') as f:
        job.dump(transformer, transformer_path)

    def _load_transformer(self, pca_id, layer_name):
        transformer_dir = '../output/transformers/'
        filename = f'pca-{pca_id}-{layer_name}'
        transformer_path = os.path.join(transformer_dir, filename)
        # with open(transformer_path, 'rb') as f:
        transformer = job.load(transformer_path)
        return transformer


class SemanticStack(Stack):
    @classmethod
    def build(cls, image_dir, model, layer_range=None, pca_dim=None,
              vector_buffer_size=100, index_buffer_size=6500, max_files=2000):
        pass

    @classmethod
    def load(cls, lib_path, layer_range=None, model=None):
        pass

    def save(self, lib_path):
        pass

    def query(self, image_path, embedding_weights=None, n_results=5,
              write_output=True):
        pass
