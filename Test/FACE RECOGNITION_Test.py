# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np


image_dir =  "/Users/Naver/Downloads/THum/01hotissue_thum.jpg"; #os.getcwd()+"/test/";

file_name_list = [image_dir]

file_name_queue = tf.train.string_input_producer(file_name_list)

reader = tf.WholeFileReader();

key, value = reader.read(file_name_queue)

image_decoded = tf.image.decode_jpeg(value)

x = tf.cast(image_decoded, tf.float32)


image_width = 330
image_height = 330

#x = tf.placeholder(tf.float32, shape=[None, image_width, image_height])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W_hidden1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
b_hidden1 = tf.Variable(tf.zeros([32]))

x_image = tf.reshape(x, shape=[-1, image_width, image_height, 1])

conv1 = tf.nn.conv2d(x_image, W_hidden1, strides=[1,1,1,1], padding="SAME")
hidden1 = tf.nn.relu(conv1+b_hidden1)

W_hidden2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
b_hidden2 = tf.Variable(tf.truncated_normal([64]))

conv2 = tf.nn.conv2d(hidden1, W_hidden2, strides=[1,1,1,1], padding="SAME")
hidden2 = tf.nn.relu(conv2+b_hidden2)

h_flat = tf.reshape(hidden2, [-1,image_width*image_height*64])
fc_w = tf.Variable(tf.truncated_normal([image_width*image_height*64,1]))
fc_b = tf.Variable(tf.zeros([1]))

print h_flat
print fc_w

fc = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.initialize_all_variables())
    image = sess.run(fc)

    #Image.fromarray(image).show()
    print image
    coord.request_stop()
    coord.join(thread)

#1시간 31분 까지 봄 , 이어서 보면됨 https://www.youtube.com/watch?v=dB6HUQMsFYchttps://www.youtube.com/watch?v=dB6HUQMsFYc