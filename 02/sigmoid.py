#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def sigmoid_prime( x ):
    y = sigmoid( x )
    return y * ( 1-y )

x = np.arange( -5 , 5 , 0.1 )
y = sigmoid( x=x )
y_prime = sigmoid_prime( x=x )

plt.grid()
plt.plot( x , y )
plt.plot( x , y_prime )
#plt.show()
plt.savefig( "./graphs/sigmoid.png" )
