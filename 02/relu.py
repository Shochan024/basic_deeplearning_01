#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def relu( x ):
    return np.where( x >= 0 , x , 0 )

def relu_prime( x ):
    return np.where( x >=0 , 1 , 0 )

x = np.arange( -2 , 2 , 0.01 )
y = relu( x=x )
y_prime = relu_prime( x=x )

plt.grid()
plt.plot( x , y )
plt.plot( x , y_prime )
plt.savefig( "./graphs/relu.png" )
