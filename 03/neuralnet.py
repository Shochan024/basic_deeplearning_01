#!-*-coding:utf-8-*-
import numpy as np

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )

def loss( y , t ):
    return 0.5 * np.sum( ( y-t )**2 )

classes = ["dog","cat"]
t = np.array([1,0])
X_0 = np.array([0.3,0.2,0.1])
W_1 = np.random.randn(64,3)
u_1 = np.dot( W_1 , X_0 )

X_1 = sigmoid( x=u_1 )
W_2 = np.random.randn( 2 , 64 )
u_2 = np.dot( W_2 , X_1 )
y = softmax( u_2 )

y_list = list( y )
result = y_list.index( max( y_list ) )

e = loss( y=y , t=t )

print( "正解:{} 予測結果:{}".format( classes[0] ,
 classes[result] ) )

print( "誤差 : {}".format( e ) )
