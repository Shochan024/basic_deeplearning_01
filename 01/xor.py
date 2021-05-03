#!-*-coding:utf-8-*-
import numpy as np

def step( x , theta=0 ):
    return np.where( x >= theta , 1 , 0 )

def XOR( x0_1 , x0_2 ):
    # 一層目
    X_1 = np.array([x0_1,x0_2])
    W_1 = np.array([[-0.2,-0.2],[0.1,0.1]])
    B_1 = np.array([0.2,-0.1])
    U_1 = np.dot( W_1 , X_1 ) + B_1

    X_2 = step( U_1 )

    # 二層目
    W_2 = np.array([0.5,0.5])
    B_2 = np.array([0])

    U_2 = np.dot( W_2 , X_2 ) #+ B_2

    return U_2

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            #print( "x_1={},x_2={} : {}".format( i , j , GATE( i , j ) ) )
            print( "x_1={},x_2={} : {}".format( i , j , step( GATE( i , j ) ,
             theta=0.8 ) ) )

"""
0 0 : 0
0 1 : 1
1 0 : 1
1 1 : 0
"""
test( XOR )
#XOR( x_1=0,x_2=0 )
