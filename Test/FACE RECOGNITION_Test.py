# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np


image_dir =  "/Users/Naver/Downloads/THum/01hotissue_thum.jpg"; #os.getcwd()+"/test/";
label_dir =  "/Users/Naver/Downloads/THum/label.csv"

imagename_dir = [image_dir]
labelname_dir = [label_dir]


imagename_queue = tf.train.string_input_producer(imagename_dir)
labelname_queue = tf.train.string_input_producer(labelname_dir)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image_decoded = tf.image.decode_jpeg(image_value)
label_decoded = tf.decode_csv (label_value, record_defaults=[[0]])

label = tf.cast(label_decoded, tf.float32)

x = tf.cast(image_decoded, tf.float32)
y_ = tf.cast(label, tf.float32)
y_ = tf.reshape(y_, [-1,1])

image_width = 330
image_height = 330

#x = tf.placeholder(tf.float32, shape=[None, image_width, image_height])
#y_ = tf.placeholder(tf.float32, shape=[None, 1])


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
fc_w = tf.Variable(tf.truncated_normal([image_width*image_height*64,10]))
fc_b = tf.Variable(tf.zeros([10]))

print h_flat
print fc_w

h_fc1 = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)

W_out = tf.Variable(tf.truncated_normal([10, 1]))
b_out = tf.Variable(tf.zeros([1]))

pred = tf.matmul(h_fc1, W_out) +b_out

print '---'
print pred
print y_

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_))
train = tf.train.AdamOptimizer(1e-1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(train)
        _cost, _accuracy = sess.run([loss, accuracy])
        print "-----------------"
        print "loss : ", _cost
        print "accuracy : ", _accuracy
    #image = sess.run(pred)

    #Image.fromarray(image).show()
    #print image
    coord.request_stop()
    coord.join(thread)

#13분 35초