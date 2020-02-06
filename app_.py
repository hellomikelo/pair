import streamlit as st
import pandas as pd
import numpy as np
from pair.utils import plot_results, save_image, get_result_images, build_montages, load_image
from pair.style_stack import StyleStack
import os
import math

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
ref_option = st.selectbox(
	'What furniture type would you like to start with?',
	('bed', 'chair', 'clock', 'couch', 'dining_table', 'plant_pot'),
	key='ref')

REF_IMG_DIR = os.path.join(DATA_ROOT, ref_option)
REF_IMG_PATHS = [os.path.join(REF_IMG_DIR, fname) for fname in os.listdir(REF_IMG_DIR)]

# display furniture choices within the group
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, use_column_width=True)

ref_id = st.text_input('Enter the number of a furniture you like', '1')
assert ref_id.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id)])
st.image(ref_img, width=300, caption='your choice')

# choose query option
lib_option = st.selectbox(
	'What furniture type would you like to get recommendations?',
	('bed', 'chair', 'clock', 'couch', 'dining_table', 'plant_pot'),
	key='query')

FEAT_LIB_PATH = os.path.join(FEAT_LIB_ROOT, 'feat_lib_'+lib_option+'/')
assert os.path.isdir(FEAT_LIB_PATH), 'Sorry! A library for this furniture does not exist yet.'

# load pre-embedded feature library
stack = StyleStack.load(FEAT_LIB_PATH);

# query based on user input
results = stack.query(REF_IMG_PATHS[int(ref_id)], None, N_RESULTS, write_output=False);    
st.subheader('Furniture recommendations')

# display furniture recommendations
query_img, results_img = get_result_images(results, N_RESULTS)
st.image(results_img, use_column_width=True, caption='recommended furnitures')

# choose which feature extractor layer to use
feat_extractor = st.slider('Feature extractor layer', 0, 5)