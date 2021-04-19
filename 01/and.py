#!-*-coding:utf-8-*-
import numpy as np

def step( x , theta=0.8 ):
    return np.where( x >= theta , 1,0 )

def AND( x_1 , x_2 ):
    x = np.array([ x_1 , x_2 ])
    w = np.array([ 0.5 , 0.5 ])
    u = np.dot( x , w )

    return u

def test( GATE ):
    for i in [0,1]:
        for j in [ 0 , 1 ]:
            print( "x_1={},x_2={} : y={}".format( i , j , step( x=GATE(i,j))))

test( GATE=AND )
