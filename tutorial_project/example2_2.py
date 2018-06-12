# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:23:49 2017

@author: hp
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBACH_SIZE = 100

data = input_data.read_data_sets('MINIST',one_hot=True)
print("hello_world")
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32,[None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))


gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.arg_max(y_pred,1),tf.arg_max(y_true,1))

accuracy = tf.reduce_mean(tf.cast(correct_mask,tf.float32))

with tf.Session() as sess:
    #train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        bach_xs, bach_ys = data.train.next_batch(MINIBACH_SIZE)
        sess.run(gd_step, feed_dict={x: bach_xs,y_true: bach_ys})
        ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                            y_true: data.test.labels})
        print("Accuracy: {:.4}%".format(ans*100))
        