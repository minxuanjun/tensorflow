# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:23:37 2018

@author: minxuan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plt.ion()
#plt.show()
BATCH_SIZE = 30
seed = 2345
rng = np.random.RandomState(seed)
X = rng.randn(300,2)

Y_ = [int(x0*x0 + x1*x1 < 2.0) for x0,x1 in X]
color = ['red' if y else 'blue' for y in Y_]


X = np.vstack(X).reshape((-1,2))
Y_ = np.vstack(Y_).reshape((-1,1))


def  get_weight(shape,regularizer):
    W = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(W))
    return W
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2)+b2


loss_mes = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mes + tf.add_n(tf.get_collection('losses'))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mes)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40001
    for i in range(STEPS):
        start  = (i*BATCH_SIZE) % 300
        end  = (start + BATCH_SIZE)%300
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mes_v = sess.run(loss_mes, feed_dict={x:X,y_:Y_})
            print("After %d steps, loss is %f"%(i, loss_mes_v))
        
            xx,yy  = np.mgrid[-3:3:0.01, -3:3:0.01]
            grid = np.c_[xx.flatten(),yy.flatten()]
            probs = sess.run(y,feed_dict={x:grid})
            plt.axis('equal') 
            plt.cla()
            plt.scatter(X[:,0], X[:,1],c=color)
            plt.contour(xx,yy,probs.reshape(xx.shape),levels=[0.5])
            plt.pause(0.001)