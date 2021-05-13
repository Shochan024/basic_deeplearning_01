#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt

def sigmoid( u ):
    return 1 / ( 1 + np.exp( -u ) )

def sigmoid_prime( u ):
    return sigmoid( u ) * ( 1 - sigmoid( u ) )

def softmax( u ):
    return np.exp( u ) / np.sum( np.exp( u ) , axis=0 )

def E( y , t ):
    return 0.5 * np.sum( ( y-t )**2 )

def E_prime( y , t ):
    return y - t

def accuracy( y , t ):
    correct_y_indexs = y.argmax( axis=0 )
    return np.sum( T.argmax( axis=0 ) == correct_y_indexs ) / y.shape[1]


# データ定義
T = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0],
    [0,1]
]).T

X1 = np.array([
    [3,0],
    [-2,5],
    [-3,1],
    [8,-1],
    [1,-5],
]).T

# 初期化
epoch = 10000
eta = 0.001
W2 = np.random.randn( 5,2 )
W3 = np.random.randn( 2,5 )

earray = []
acc_array = []
for i in range( epoch ):
    # 順伝播
    z1 = np.dot( W2 , X1 )
    X2 = sigmoid( u=z1 )

    z2 = np.dot( W3 , X2 )
    y = softmax( u=z2 )

    e = E( y=y , t=T )
    acc = accuracy( y=y , t=T )

    earray.append( e )
    acc_array.append( acc )

    # 逆伝播
    dout = E_prime( y=y,t=T )
    grad_W3 = np.dot( X2 , dout.T )

    dhidden = np.dot( W3.T , dout ) * sigmoid_prime( u=z1 )
    grad_W2 = np.dot( X1 , dhidden.T )

    W2 -= eta * grad_W2.T
    W3 -= eta * grad_W3.T

plot_x = np.arange( 0 , epoch , 1 )

fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot( 211 , xlabel="epoch" , ylabel="error func value" )
ax2 = fig.add_subplot( 212 , xlabel="epoch" , ylabel="Accuracy" )

ax1.plot( plot_x , earray )
ax1.grid( c="gainsboro" , zorder=9 )

ax2.plot( plot_x , acc_array )
ax2.grid( c="gainsboro" , zorder=9 )
fig.savefig( "graphs/loss_and_accuracy.png" )
