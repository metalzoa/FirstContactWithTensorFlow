
## Hands-on: First Contact With TensorFlow
This repository contains the support material for the TensorFlow Hands-on at [Summer Seminar ETSETB TelecomBCN, 4-8 July 2016 (http://telecomBCN.DeepLearning.Barcelona)] (http://telecomBCN.DeepLearning.Barcelona).



#### Documentation

We are using the book [First Contact with TensorFlow] (http://www.jorditorres.org/first-contact-with-tensorflow-book/) 
as a basic documentation. You can acces a [freely available on-line copy] (http://www.jorditorres.org/first-contact-with-tensorflow/>).

We assume that the student has some basic knowledge about Python. If not, a Python Quick Start hands-on that will help to start with this language can be found [here (Python Quick Start)](http://www.jorditorres.org/teaching-activity/hands-on-1-python-quick-start/).


#### TensorFlow installation (do it before the course starts)
For the labs, you should have a working installation of Python. TensorFlow has a Python API (plus a C / C ++) that requires the installation of Python 2.7. Nowadays many Linux and UNIX distributions include a recent Python.If this is not the case I assume that any student who take this course knows how to install it from the [general download page]( https://www.python.org/downloads/). 

We will use a virtual environment virtualenv to install TensorFlow (this will not overwrite existing versions of Python packages from other projects required by TensorFlow).

First, you should install pip and virtualenv if they are not already installed, like the follow script shows:
```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv 

# Mac OS X 
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```
environment virtualenv in the ~/tensorflow directory:

```
$ virtualenv --system-site-packages ~/tensorflow
```

The next step is to activate the virtualenv. This can be done as follows:

```
$ source ~/tensorflow/bin/activate #  with bash 
$ source ~/tensorflow/bin/activate.csh #  with csh
(tensorflow)$
```
The name of the virtual environment in which we are working will appear at the beginning of each command line from now on. Once the virtualenv is activated, you can use pip to install TensorFlow inside it:

```
# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl 

# Mac OS X, CPU only:
(tensorflow)$ sudo easy_install --upgrade six
(tensorflow)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```
In order to be sure that everything is properly working create a simple TensorFlow code and save it with extension ".py". I suggest the following code

```
import tensorflow as tf
  
 a = tf.placeholder("float")
 b = tf.placeholder("float")
  
 y = tf.mul(a, b)
  
 sess = tf.Session()
  
 print sess.run(y, feed_dict={a: 3, b: 3})
```
To run the code, it will be enough with the command 

```
$ python test.py
```
If the result is 9, it means that TensorFlow is proferly installed.

Finally, when you’ve finished, you should disable the virtual environment as follows:

```
(tensorflow)$ deactivate
```

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











Installation
============
