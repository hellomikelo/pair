import streamlit as st
import pandas as pd
import numpy as np
from pair.util import plot_results, save_image, get_result_images, build_montages, load_image
from pair.style_stack import StyleStack
import os
import math
import random

import tensorflow.keras.applications as apps

st.title('Pair demo: IKEA furniture collection recommender')
st.markdown(
"""
Pair is an image-based product collection recommender that pair user-preferred products
with other visually compatible product collections. This is a demo to showcase how Pair works.
Use the slider to select different design feature extractors, and use the checkbox to see 
which furniture you'd like to see recommendations for.
""")

# initial parameters
DATA_ROOT = './data/'
FEAT_LIB_ROOT = './output/indexes/'
N_RESULTS = 4
MONTAGE_COL = 5

# choose ref image category
st.sidebar.subheader('Pick what you like')
ref_option = st.sidebar.selectbox(
	'Choose furniture type:',
	('bed', 'chair', 'clock', 'couch', 'dining_table', 'plant_pot'),
	key='ref')

REF_IMG_DIR = os.path.join(DATA_ROOT, ref_option)
ALL_REF_IMG_PATHS = [os.path.join(REF_IMG_DIR, fname) for fname in os.listdir(REF_IMG_DIR)]
# REF_IMG_PATHS = random.sample(ALL_REF_IMG_PATHS, 10)
REF_IMG_PATHS = ALL_REF_IMG_PATHS[:10]

# display furniture choices within the group
st.subheader('Furniture choices')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, use_column_width=True)

# img_choice = st.sidebar.empty()

ref_id = st.sidebar.text_input('Enter preferred furniture ID', '1')
assert ref_id.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id)])
st.sidebar.image(ref_img, use_column_width=True)
# img_choice.image(ref_img, width=300, caption='your choice')

# choose query option
lib_option = st.sidebar.selectbox(
	'Choose furniture recommendation type:',
	('bed', 'chair', 'clock', 'couch', 'dining_table', 'plant_pot'),
	key='query',
	index=4)

model = apps.vgg16.VGG16(weights='imagenet', include_top=False)
st.text(type(model))

FEAT_LIB_PATH = os.path.join(FEAT_LIB_ROOT, 'feat_lib_'+lib_option+'/')
assert os.path.isdir(FEAT_LIB_PATH), 'Sorry! A library for this furniture does not exist yet.'

# load pre-embedded feature library
stack = StyleStack.load(FEAT_LIB_PATH);

# query database based on user input
results = stack.query(
	image_path=REF_IMG_PATHS[int(ref_id)], 
	lib_path=FEAT_LIB_PATH,
	embedding_weights=None, 
	n_results=N_RESULTS, 
	write_output=False);    

st.subheader('Pair recommendations')
# display furniture recommendations
query_img, results_img = get_result_images(results, N_RESULTS)
st.image(results_img, use_column_width=True)

# choose which feature extractor layer to use
# feat_extractor = st.slider('Feature extractor layer', 0, 5)