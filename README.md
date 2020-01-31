# ai-curator
Deep Style Curator is an image-based product recommender for finding visually similar furnitures across different categories. The user provides an image of a furniture he or she likes, and the model will return a series of furnitures from other categories that are similar in design. In short, the model uses a convolutional neural network (VGG16) to learn and build a library of designs of collections of furnitures. Then when a new image is introduced, the model queries the new image  with the library to recommend other products of similar design that the user may like.

This repository contains two main modules:
1. __Recommender module__: It generates design feature extractors and makes furniture recommendations based on user input images
2. __Trainer module__: [in-progress] It performs transfer learning on a pre-trained VGG16 network to learn to classify furnitures based on the type of rooms they may appear in. The purpose is to improve the design feature extractors so it can better recognize complementary furnitures

## Credits
This project is built on top of Austin McKay's [Style Stack repo](https://github.com/TheAustinator/style-stack). It is also inspired by Greg Surma's [Style Transfer repo](https://github.com/gsurma/style_transfer/blob/master/style-transfer.ipynb) and Ivona Tautkute et al.'s [IKEA image dataset repo](https://github.com/yuanhunglo/ikea).

Short explanation of the directories:

* `_prototype_style_stack` contains prototype of the recommender system based on Austin McKay's Style Stack
* `_prototype_trainer` contains prototype files to be submitted to Google Cloud Platform for training
* `build` contains web page related files
* `data` contains small amounts of prototyping data
* `output` contains the resulting model output images
* `tests` contains model tests

## Recommender module

### Installation
The model is tested on Python 3.6. Dependencies are listed in `requirements.txt`. To install these Python dependencies, please run 
> `pip install -r requirements.txt` 

Or if you prefer to use conda, 
> `conda install --file requirements.txt`

### Usage

To run the prototyping and get an inference, please go to `_prototype_style_stack/` and simply run:

> `python run_model.py`

If you'd like to play around with different images, please only change parameters in `configs.py` and run `run_model.py`.

The prototype will take as input path to an image of a table and the model will recommend different chairs based on the table's style and content. Result image will be saved in the `output/` directory. For prototyping, inference time is on the order of seconds. 

## Transfer learning module

### Installation 
To build the development environment from a Docker image (takes \~5 minutes), run
> `docker build -t <image-name> .`

To run the Docker image interactively from the shell, run
> `docker run -i -t <image-name> /bin/bash`

To run transfer learning module, run
> `python -m trainer.task --data-dir './data/' --out-dir './output/'`


### Dataset

A small dataset from Ivona Tautkute et al. ([IKEA image dataset](https://github.com/yuanhunglo/ikea)) is used to create the minimal viable product.

