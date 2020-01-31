import sys
from numpy import load

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend

import pickle
import os
import pandas as pd
import yaml
import tensorflow as tf


def load_configs(config_path):
	with open(config_path, 'r') as stream:
	    try:
	        configs = yaml.safe_load(stream)
	        print(yaml.safe_load(stream))
	    except yaml.YAMLError as exc:
	        print(exc)

def load_classes(args):
	room_names = os.listdir(os.path.join(args.data_dir, 'room_scenes/'))
	# room_names = tf.gfile.ListDirectory(dir_path)

	# match path string to input paths (not actually loading images)
	classes = [args.data_dir + 'room_scenes/' + x for x in room_names]
	return classes
 
def make_labels_df(args):
	""" make dataframe for ImageDataGenerator """
	f = open(os.path.join(args.data_dir, 'text_data/item_to_room.p'), 'rb')
	# f = tf.gfile.Open(os.path.join(args.job_dir, 'text_data/item_to_room.p'), 'rb')
	data = pickle.load(f)
	data = {key: [value] for key, value in data.items()}
	# make list of all room names
	df = pd.DataFrame.from_dict(data, orient='index').reset_index()
	df.columns = ['file_name', 'room_name']
	# clean up file names
	df['room_name'] = df['room_name'].map(lambda x: [args.data_dir + line.split('/', 1)[1] for line in x])
	df['file_name'] = df.file_name.apply(lambda x: x.split('/')[-1])
	return df


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
 
 
# plot diagnostic learning curves
def summarize_diagnostics(history, export_path):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Fbeta')
	pyplot.plot(history.history['fbeta'], color='blue', label='train')
	pyplot.plot(history.history['val_fbeta'], color='orange', label='test')
	# save plot to file
	# filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(export_path)
	pyplot.close()

# # load train and test dataset
# def load_dataset():
# 	# load dataset
# 	data = load('planet_data.npz')
# 	X, y = data['arr_0'], data['arr_1']
# 	# separate into train and test datasets
# 	trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=1)
# 	print(trainX.shape, trainY.shape, testX.shape, testY.shape)
# 	return trainX, trainY, testX, testY
