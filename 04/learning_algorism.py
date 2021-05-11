#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

def scatter( rho , n ):
    x = np.random.randn( n )
    y = x * rho + np.sqrt( 1 - rho**2 ) * np.random.randn( n )

    return x , y

def loss( y , t ):
    return 0.5 * np.sum( ( y - t ) ** 2 )

def get_grad( X , B , T ):
    Xt_X = np.dot( X.T , X )
    Xt_X_B = np.dot( Xt_X , B ).T
    Xt_T = np.dot( X.T , T )

    grad = Xt_X_B - Xt_T

    return grad.T

def approximation_line( a , b , x ):
    return a*x+b


# 初期化
n = 100
epoch = 1000
eta = 0.001
rho = 0.7
X , T = scatter( rho=rho , n=n )
intercept = np.ones_like( X )
B = np.random.randint( -10 , 10 , ( 2 , 1 ) )

# 説明変数行列の整形
X = X.reshape( n , 1 )
intercept = intercept.reshape( n , 1 )
X = np.append( X , intercept , axis=1 )

approximation_line_x = np.arange( -4 , 4 , 1 )

E = []
B_values = []
ims = []
fig = plt.figure()
plt.grid()
plt.scatter( X[:,0] , T )
for i in range( epoch ):
    y_hat = approximation_line( a=B[0] , b=B[1] ,
     x=approximation_line_x )

    ims.append( plt.plot( approximation_line_x ,
     y_hat , color="orange" ) )

    # 推定( 説明変数と係数ベクトルの内積 )
    Y = np.dot( X , B )
    e = loss( y=Y , t=T )

    B_values.append( B[0] )
    E.append( e )

    # 回帰パラメータの更新
    grad = get_grad( X=X , B=B , T=T )
    B = B - eta * grad


#######################
#      Animation      #
#######################
anim = ArtistAnimation( fig , ims , interval=1 , blit=True )
anim.save( "graphs/fitting.gif" )


#######################
#     Transitions     #
#######################
plot_x = np.arange( 0 , len( E ) , 1 )
fig2 = plt.figure( tight_layout=True )
ax1 = fig2.add_subplot( 211 , ylim=(-10,10) ,
 title="coefficient transition" )

ax2 = fig2.add_subplot( 212 , title="error function value transition" )

ax1.grid()
ax2.grid()

ax1.plot( plot_x , B_values )
ax1.plot( plot_x , np.ones_like( plot_x ) * rho ,
 linestyle="dashed" , c="black" )

ax2.plot( plot_x , E )

fig2.savefig("graphs/fitting_transition.png")
