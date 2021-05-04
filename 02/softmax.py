#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )

X = np.array([1,2,3])
#print( np.exp( x ) )
#print( np.exp( x ) )
print( softmax( x=X ) )
