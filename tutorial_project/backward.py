# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:11:59 2018

@author: minxuan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_generate
import forward

STEPS  = 40000
BATCH_SIZE = 30
LEARN_RATEING_BASE = 0.001
LEARN_RATEING_DECAY = 0.999
REGULARIZER = 0.001


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_= tf.placeholder(tf.float32, shape=(None,1)) 
    
    X, Y_, color = data_generate.generateData()
    
    y = forward.forward(x,REGULARIZER)
    
    global_step = tf.Variable(0,trainable=False)
    learn_rating = tf.train.exponential_decay(LEARN_RATEING_BASE
                                              ,global_step
                                              ,300/BATCH_SIZE
                                              ,LEARN_RATEING_DECAY
                                              ,staircase=True)
    #define loss function
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
    #define backward 
    train_step = tf.train.AdamOptimizer(learn_rating).minimize(loss_total)
    loss_array = []
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        for i in range(STEPS+1):
            start = (i*BATCH_SIZE)% 300
            end = (start + BATCH_SIZE)%300
            sess.run(train_step,feed_dict={x: X[start:end],y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total,feed_dict={x:X,y_:Y_})
                loss_array.append(loss_v)
                print("After %d steps, loss is %f"%(i, loss_v))
                xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
                grid = np.c_[xx.flatten(),yy.flatten()]
                probs = sess.run(y, feed_dict={x:grid})
                probs = probs.reshape(xx.shape)
                plt.figure(1)
                plt.cla()
                plt.axis('equal')
                plt.scatter(X[:,0],X[:,1],c = color)
                plt.contour(xx, yy, probs,levels=[0.5])
                plt.pause(0.001)
                plt.figure(2)
                plt.cla()
                plt.plot(loss_array)
                plt.pause(0.001)
                
if __name__ == '__main__':
    backward()