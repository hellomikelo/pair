import argparse
import pickle
import os
import pandas as pd

from . import model
from . import util

import sys
from numpy import load
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf



""" 
Transfer learning using VGG16 to train a furniture collection classifier. 
Freeze first 4 conv layers and only train on last conv and full-nected layers

codes modified from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/ 
and from https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24

Mike Lo
"""

def get_args():
	"""Argument parser.

	Returns:
	Dictionary of arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		default='./data/',
		type=str,
		help='default data directory')
	parser.add_argument(
		'--out-dir',
		default='./output/',type=str,
		help='default output save directory')  
	parser.add_argument(
		'--cpt-path',
		default='./output/weights.best.hdf5',
		type=str,
		help='checkpoint path')  	
	parser.add_argument(
		'--num-epochs',
		type=int,
		default=2,
		help='number of times to go through the data, default=2')
	parser.add_argument(
		'--batch-size',
		default=10,
		type=int,
		help='number of records to read during each training step, default=128')  
	parser.add_argument(
		'--learning-rate',
		default=.01,
		type=float,
		help='learning rate for gradient descent, default=.01')
	parser.add_argument(
		'--verbosity',
		choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
		default='INFO')
	args, _ = parser.parse_known_args()
	return args

# run the test harness for evaluating a model
def train_and_evaluate(args):

	# load the filenames of all room_scenes as classes for classification
	# classes = util.load_classes('./data/room_scenes')
	classes = util.load_classes(args)

	# load data labels
	df_labels = util.make_labels_df(args)

	# create data generator
	train_datagen = ImageDataGenerator(featurewise_center=True, horizontal_flip=True, vertical_flip=True, rotation_range=90)
	test_datagen = ImageDataGenerator(featurewise_center=True)

	# specify imagenet mean values for centering
	train_datagen.mean = [123.68, 116.779, 103.939]
	test_datagen.mean = [123.68, 116.779, 103.939]
	
	# prepare iterators
	# train_it = train_datagen.flow(trainX, trainY, batch_size=128)
	# test_it = test_datagen.flow(testX, testY, batch_size=128)

	train_it = train_datagen.flow_from_dataframe(
	    dataframe=df_labels[0:100],
	    directory=os.path.join(args.data_dir, 'images'),
	    x_col="file_name",
	    y_col="room_name",
	    batch_size=args.batch_size,
	    seed=42,
	    shuffle=True,
	    classes=classes,
	    class_mode="categorical",
	    target_size=(128, 128))

	test_it = test_datagen.flow_from_dataframe(
	    dataframe=df_labels[100:200],
	    directory=os.path.join(args.data_dir, 'images'), 
	    x_col="file_name",
	    y_col="room_name",
	    batch_size=args.batch_size,
	    seed=42,
	    shuffle=True,
	    classes=classes,
	    class_mode="categorical",
	    target_size=(128, 128))

	# define model
	print('> Define model')
	keras_model = model.create_model()

	# fit model
	print('> Begin training')
	print('> len(train_it) = {}'.format(len(train_it)))
	print('> len(test_it) = {}'.format(len(test_it)))

	# create checkpoints
	# filepath=os.path.join(args.cpt_path, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
	# checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max', period=2)
	# # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	# callbacks_list = [checkpoint]	

	history = keras_model.fit_generator(train_it, 
		steps_per_epoch=len(train_it),
		validation_data=test_it, 
		validation_steps=len(test_it), 
		epochs=args.num_epochs, 
		verbose=1)

	# evaluate model
	print('> Begin evaluating')
	loss, fbeta = keras_model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
	print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))
	
	# save model
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)  
	keras_model.save(os.path.join(args.out_dir, 'final_model.h5'))
	
	# learning curves (doesn't work on headless server, i.e. GCP)
	util.summarize_diagnostics(history, os.path.join(args.out_dir, 'diagnostics.png'))
	
	# model.load_weights("weights.best.hdf5")
	# # Compile model (required to make predictions)
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print("Created model and loaded weights from file")
 
if __name__ == '__main__':
	args = get_args()
	train_and_evaluate(args)