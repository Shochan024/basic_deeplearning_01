import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.datasets import cifar10

history_file = "./histories/cifar10.csv"
model_file = "./models/cifar10.h5"
model_yaml_file = "./models/cnn_model.yaml"

##############################
#        訓練課程の可視化      #
##############################

df = pd.read_csv( history_file )
epoch = len( df )
fig1 = plt.figure( tight_layout=True )
plot_x = np.arange( 0 , epoch , 1 )

ax1 = fig1.add_subplot( 211 , xlabel="epoch" , ylabel="loss" )
ax2 = fig1.add_subplot( 212 , xlabel="epoch" , ylabel="Accuracy" )

ax1.grid()
ax1.plot( plot_x , df["loss"] , label="train_loss" )
ax1.plot( plot_x , df["val_loss"] , label="val_loss" )
ax1.legend()

ax2.grid()
ax2.plot( plot_x , df["accuracy"] , label="train_accuracy" )
ax2.plot( plot_x , df["val_accuracy"] , label="val_accuracy" )
ax2.legend()
plt.savefig( "graphs/model_specs.png" )

###############################################
#    外部ファイルから学習済みモデルの読み込み       #
###############################################

labels = ["airplane","autmobile","bird","cat","deer","dog","frog","horse","ship","truck"]

( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

with open( model_yaml_file , "r" ) as yaml_file:
    yaml_string = yaml_file.read()

model = model_from_yaml( yaml_string )
model.load_weights( model_file )


##############################
#        分類結果の可視化       #
##############################
axes = []
fig2 = plt.figure( tight_layout=True )
for i in range(10):
    index = np.random.randint( 0 , X_test.shape[0] )
    img = X_test[index]
    correct_label =  labels[ y_test[index][0] ]
    result = model.predict( img.reshape( 1 , 32 , 32 , 3 ) )
    axes.append( fig2.add_subplot( 2 , 5 , i+1 ,
     xlabel="c:{} p:{}".format( correct_label , labels[ result.argmax() ] ) ) )

    axes[i].imshow( img )
plt.savefig( "graphs/predicts.png" )


##############################
#      複数画像一括処理の例     #
##############################
result_test = model.predict( X_test )
print( result_test.argmax( axis=1 ) )
