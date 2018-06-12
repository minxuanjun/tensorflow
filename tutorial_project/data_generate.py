# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:49:55 2018

@author: minxuan
"""
import numpy as np
import matplotlib.pyplot as plt

SEED = 2

def generateData():
    rng = np.random.RandomState(SEED)
    X = rng.randn(300,2)
    Y_ = [int(x0*x0 + x1*x1 <2) for (x0, x1) in X]
    color = ['red' if y else 'blue' for y in Y_]
    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)
    
    return X,Y_,color
if __name__ =='__main__':
    X,Y_,color = generateData()
    plt.scatter(X[:,0],X[:,1],c=color)
    
