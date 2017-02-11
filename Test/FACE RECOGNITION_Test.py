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

# ------------------------------------------------------------
hidden1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
hidden1_b = tf.Variable(tf.zeros([32]))

hidden2_w = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
hidden2_b = tf.Variable(tf.truncated_normal([64]))

fc_w = tf.Variable(tf.truncated_normal([image_width*image_height*64,10]))
fc_b = tf.Variable(tf.zeros([10]))

out_w = tf.Variable(tf.truncated_normal([10, 1]))
out_b = tf.Variable(tf.zeros([1]))
# --------------------------------------------------------------
x_image = tf.reshape(x, shape=[-1, image_width, image_height, 1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, hidden1_w, strides=[1, 1, 1, 1], padding="SAME") + hidden1_b)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, hidden2_w, strides=[1, 1, 1, 1], padding="SAME") + hidden2_b)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

h_flat = tf.reshape(h_pool2, [-1, image_width * image_height * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)

drop_fc = tf.nn.dropout(h_fc1, 0.5)

pred = tf.matmul(drop_fc, out_w) + out_b

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------------------s
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

#1시간 40분 20초