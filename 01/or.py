#-*-coding:utf-8-*-
import numpy as np

def step( x , theta=0.3 ):
    return np.where( x >= theta , 1 , 0 )


def OR( x_1 , x_2 ):
    x = np.array([ x_1 , x_2 ])
    w = np.array([ 0.5 , 0.5 ])
    u = np.dot( x , w )

    return u

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            z = GATE( x_1=i , x_2=j )
            y = step( x=z )
            print( "x_1={},x_2={} : y={}".format( i , j , y ) )


test( GATE=OR )
