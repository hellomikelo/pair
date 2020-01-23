# Deep Style Curator
Deep Style Curator is an image-based product recommender system for finding visually similar furnitures across different categories. In short, it uses a convolutional neural network to learn the style and content of furnitures and match them with other similar products that the user may like.

This project is built on top of Austin McKay's [Style Stack repo](https://github.com/TheAustinator/style-stack). It is also inspired by Greg Surma's [Style Transfer repo](https://github.com/gsurma/style_transfer/blob/master/style-transfer.ipynb) and Ivona Tautkute et al.'s [IKEA image dataset repo](https://github.com/yuanhunglo/ikea).

Short explanation of the directories:

* `_prototype_style_stack` contains prototype of the recommender system based on Austin McKay's Style Stack
* `_prototype_trainer` contains prototype files to be submitted to Google Cloud Platform for training
* `build` contains web page related files
* `data` contains small amounts of prototyping data
* `output` contains the resulting model output images
* `tests` contains model tests

## Installation

The model is tested on Python 3.6 and has the following dependencies:  
* keras  
* sklearn  
* scipy  
* matplotlib  
* numpy  
* pillow  
* scikit-learn  
* faiss  

To install these Python dependencies, please run 
> `pip install -r requirements.txt` 

Or if you prefer to use conda, 
> `conda install --file requirements.txt`

There are plans to containerize the environment using Docker (TBA).

## Usage

To run the prototyping and get an inference, please go to `_prototype_style_stack/` and simply run:

> `python run_model.py`

If you'd like to play around with different images, please only change parameters in `configs.py` and run `run_model.py`.

The prototype will take as input path to an image of a table and the model will recommend different chairs based on the table's style and content. Result image will be saved in the `output/` directory. For prototyping, inference time is on the order of seconds. 

## Dataset

For creating a minimal viable product, Ivona Tautkute et al.'s [IKEA image dataset](https://github.com/yuanhunglo/ikea) is used.

