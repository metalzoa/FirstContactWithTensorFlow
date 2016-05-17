

This repository contains the support material for the TensorFlow Hands-on at [Summer Seminar ETSETB TelecomBCN, 4-8 July 2016 (http://telecomBCN.DeepLearning.Barcelona)] (http://telecomBCN.DeepLearning.Barcelona).


Code examples for some popular machine learning algorithms, using TensorFlow library. This tutorial is designed to easily dive into TensorFlow, through examples. It includes both notebook and code with explanations.

## Tutorial index

#### 1 - Introduction
- Hello World ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1%20-%20Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1%20-%20Introduction/helloworld.py))
- Basic Operations ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1%20-%20Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1%20-%20Introduction/basic_operations.py))

#### 2 - Basic Classifiers
- Nearest Neighbor ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/nearest_neighbor.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/nearest_neighbor.py))
- Linear Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/linear_regression.py))
- Logistic Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/logistic_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/logistic_regression.py))

#### 3 - Neural Networks
- Multilayer Perceptron ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/multilayer_perceptron.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py))
- Convolutional Neural Network ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/convolutional_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/convolutional_network.py))
- AlexNet ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/alexnet.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/alexnet.py))
- Reccurent Network ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py))

#### 4 - Multi GPU
- Basic Operations on multi-GPU ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4%20-%20Multi%20GPU/multigpu_basics.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4%20-%20Multi%20GPU/multigpu_basics.py))

#### 5 - User Interface (Tensorboard)
- Graph Visualization ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5%20-%20User%20Interface/graph_visualization.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5%20-%20User%20Interface/graph_visualization.py))
- Loss Visualization ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5%20-%20User%20Interface/loss_visualization.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5%20-%20User%20Interface/loss_visualization.py))

## Dependencies
```
tensorflow
numpy
matplotlib
cuda (to run examples on GPU)
```
For more details about TensorFlow installation, you can check [Setup_TensorFlow.md](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/Setup_TensorFlow.md)

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples (with input_data.py).
MNIST is a database of handwritten digits, with 60,000 examples for training and 10,000 examples for testing. (Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))








Course: First Contact with TensorFlow
******
This repository contains the support material for the  `Summer Seminar ETSETB TelecomBCN, 4-8 July 2016 (http://telecomBCN.DeepLearning.Barcelona) <http://telecomBCN.DeepLearning.Barcelona>`.


Documentation
=============
We are using the book `First Contact with TensorFlow <http://www.jorditorres.org/first-contact-with-tensorflow-book/>`_ 
as a basic documentation. You can acces a `freely available on-line copy <http://www.jorditorres.org/first-contact-with-tensorflow/>`_ .


Prerequisite
============

In general, when you are working in Python, you should use the virtual environment virtualenv. Virtualenv is a tool to keep Python dependencies required in different projects, in different parts of the same computer. If we use virtualenv to install TensorFlow, this will not overwrite existing versions of Python packages from other projects required by TensorFlow.

For the labs, you should have a working installation of Python. TensorFlow has a Python API (plus a C / C ++) that requires the installation of Python 2.7. I assume that any student who take this course knows how to do it. 


Installation
============
