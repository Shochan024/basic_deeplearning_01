#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def softplus( x ):
    return np.log( 1 + np.exp( x ) )

def mish( x ):
    return x * np.tanh( softplus( x=x ) )

def mish_prime( x ):
    tanh = np.tanh( softplus( x ) )
    return x*sigmoid(x) * ( 1 - tanh**2 ) + tanh

x = np.arange( -5 , 5 , 0.1 )
y = mish( x=x )
y_prime = mish_prime( x=x )

plt.grid()
plt.plot(x,y)
plt.plot( x,y_prime )
plt.savefig( "./graphs/mish.png" )
