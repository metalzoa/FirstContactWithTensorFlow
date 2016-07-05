
# Course First Contact With TensorFlow
This course, as part of [Summer Seminar ETSETB TelecomBCN, 4-8 July 2016 (http://telecomBCN.DeepLearning.Barcelona)] (http://telecomBCN.DeepLearning.Barcelona) is basically a hands-on tutorial that provides a quick start to building applications using TensorFlow. and we will teach the esential ideas of Tensorflow ecosystem.
## Course details

#### Instructors

- [Maurici Yagües](https://bsc.es/about-bsc/staff-directory/yagues-maurici), Research engineer at BSC-CNS
- [Jordi Torres](http://www.JordiTorres.Barcelona), Professor at UPC and Researcher at BSC-CNS

#### Prerequisites

We assume that the student has some basic knowledge about Python. If not, a Python Quick Start hands-on that will help to start with this language can be found [here (Python Quick Start)](http://www.jorditorres.org/teaching-activity/hands-on-1-python-quick-start/).

We assume that the student has a `Ubuntu/Linux 64-bit` or `Mac OS X`  environment. If the student have a `Windows` environment we suggest to use `VirtualBox` in order to run a Linux in a separate virtual machine. You can follow the hands-on *RUN A LINUX OS IN A VIRTUAL MACHINE* to install it [from this web](http://www.jorditorres.org/teaching-activity/hands-on-0-run-a-linux-os-in-a-virtual-machine/).

#### Grading
Course grade are comprised of 3 homework assignments by groups (30%), class exercises (20%) and individual class attendance (50%).

#### Schedule (tentative)

##### Day 1 (Monday 04/07/2016) 

- How to build basic TensorFlow graphs and how to train models
- Case study: Linear Regression in TensorFlow

##### Day 2 (Tuesday 05/07/2016) 

- Basic data structures in TensorFlow
- Case study: Clustering in TensorFlow 

##### Day 3 (Wednesday 06/07/2016)  

- Single Layer Neural Network in TensorFlow
- TensorBoard 

##### Day 4 (Thursday 07/07/2016)  

- Convolutional Neural Networks in TensorFlow
- TensorFlow High Level APIs: SLIM

##### Day 5 (Friday 08/07/2016)  

- Recurrent Neural Networks in TensorFlow



#### Documentation/Textbook

We will use the book [First Contact with TensorFlow] (http://www.jorditorres.org/first-contact-with-tensorflow-book/) 
as a basic documentation. You can acces a [freely available on-line copy] (http://www.jorditorres.org/first-contact-with-tensorflow/>). The slides used during the hands-on will be also available before start the course. Additional documentation will be distributed during the course.

The slides and codes used during the sessions will be posted/updated 2 hours before the session:

##### Day 1

- [SLIDES (pdf)](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/TF.course.slides.day1.pdf)
- [multiplication.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/multiplication.py) 
- [regression.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/regression.py) 

##### Day 2

- [SLIDES (pdf)](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/TF.course.slides.day2.pdf)
- [clustering.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/clustering.py) 
- [multiGPU.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/MultiGPU.py) 

##### Day 3

- Slides
- [SingleLayerNeuralNetwork.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/SingleLayerNeuralNetwork.py)
- [input_data.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/input_data.py)

##### Day 4

- Slides
- [MultiLayerNeuralNetwork.py](https://github.com/jorditorresBCN/FirstContactWithTensorFlow/blob/master/MultiLayerNeuralNetworks.py)

##### Day 5
- Slides


## Installation instructions (do it before the course starts)
For the sessions, please bring your laptop, and you should have a working installation of Python. TensorFlow has a Python API (plus a C / C ++) that requires the installation of Python 2.7. Nowadays many Linux and UNIX distributions include a recent Python.If this is not the case I assume that any student who take this course knows how to install it from the [general download page]( https://www.python.org/downloads/). 

During the sessions lab the instructor could use IPython/Jupyter. If you are interested to use too, you can obtain it from [here] (https://ipython.org) (optional).

#### Virtual environment-based installation
We will use a virtual environment `virtualenv`, a tool to create isolated Python environments to install TensorFlow. This will not overwrite existing versions and dependencies (and indirectly permissions) of Python packages from other projects required by TensorFlow in your laptop.  Virtualenv creates an environment that has its own installation directories, that doesn’t share libraries with other virtualenv environments (and optionally doesn’t access the globally installed libraries either).

First, you should install `pip` and `virtualenv` if they are not already installed, like the follow script shows:
```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv 

# Mac OS X 
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```
`virtualenv` has one basic command: virtualenv. We suggest to create telecomBCN directory to place the new virtual environment for this hads-on

```
$ virtualenv --system-site-packages ~/telecomBCN
```

The python in your new virtualenv `~/telecomBCN` directory is effectively isolated from the python that was used to create it. In a newly created virtualenv there will also be a activate shell script. The next step is to activate the virtualenv. This can be done as follows:

```
$ source ~/telecomBCN/bin/activate #  with bash 
$ source ~/telecomBCN/bin/activate.csh #  with csh
```
The activate script will also modify your shell prompt to indicate which environment is currently active (will appear at the beginning of each command line from now on). In our case
```
(telecomBCN)$
```
If you are interested for more details about virtualenv you can obtain it from [here] (https://pypi.python.org/pypi/virtualenv).

#### Install TensorFlow
Once the virtualenv is activated, you can use pip to install TensorFlow inside it:

```
# Ubuntu/Linux 64-bit, CPU only:
(telecomBCN)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl 

# Mac OS X, CPU only:
AQUEST CREC QUE NO CAL --> (telecomBCN)$ sudo easy_install --upgrade six
(telecomBCN)$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl
```

The exemples in this hands-on will require install the following packages: 

```
$ sudo pip install numpy
$ sudo pip install matplotlib
```

#### My first code
In order to be sure that everything is working fine, create a simple TensorFlow code and save it with extension ".py". I suggest to use the following code `multiplication.py` from the course github:

```
import tensorflow as tf
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.mul(a, b)
sess = tf.Session()
print sess.run(y, feed_dict={a: 3, b: 3})
```
You can download it from the github using the git command: 
```
(telecomBCN)$ cd telecomBCN
(telecomBCN)$ git clone https://github.com/jorditorresBCN/FirstContactWithTensorFlow.git
```
To run the code, it will be enough with the command 

```
(telecomBCN)$ cd FirstContactWithTensorFlow
(telecomBCN)$ python multiplication.py
```
If the result is `9.0`, it means that TensorFlow is properly installed.

#### Disable the virtual environment
Finally, when you’ve finished, you should disable the virtual environment as follows:

```
(telecomBCN)$ deactivate
```




