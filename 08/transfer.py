#!-*-coding:utf-8-*-
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import resize_images
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense , Activation , Flatten , Lambda , Dropout
from tensorflow.keras.layers import BatchNormalization

####################################
#           学習データの準備         #
####################################

( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical( y_train , 10 )
y_test = to_categorical( y_test , 10 )


####################################
#           モデルの定義             #
####################################
vgg16 = VGG16( weights = "imagenet" , include_top=False , input_shape=( 224,224,3 ) )
vgg16.trainable = False

model = Sequential()
model.add(
    Lambda(
        lambda x: resize_images( x , 7 , 7 , "channels_last" ),
        input_shape=( 32,32,3 ),trainable=False
    )
)
model.add( vgg16 )
model.add( Flatten( trainable=False ) )
model.add( Dense( 4096 ) )
model.add( Activation( "relu" ) )
model.add( BatchNormalization() )
model.add( Dropout( 0.3 ) )

model.add( Dense( 4096 ) )
model.add( Activation( "relu" ) )
model.add( BatchNormalization() )
model.add( Dropout( 0.3 ) )

model.add( Dense( 10 ) )
model.add( Activation( "softmax" ) )

####################################
#        学習アルゴリズムの用意        #
####################################
learning_date = 0.001
epochs = 40
batch_size=128

model_yaml_file = "/content/drive/MyDrive/basic_deeplearning/models/imagenet2cifar10.yaml"
model_file = "/content/drive/MyDrive/basic_deeplearning/models/imagenet2cifar10.h5"
history_file = "/content/drive/MyDrive/basic_deeplearning/histories/imagenet2cifar10.csv"

if os.path.isfile( model_file ):
    model.load_weights( model_file )

adam = Adam( learning_rate=learning_date )
model.compile( loss="categorical_crossentropy" , optimizer=adam , metrics=["accuracy"] )

history = model.fit(
    X_train,
    y_train,
    validation_data=( X_test , y_test ) ,
    epochs=epochs,
    batch_size=batch_size
)

model.save_weights( model_file )

df_history = pd.DataFrame( history.history )

if os.path.isfile( history_file ):
    df = pd.read_csv( history_file )
    df = pd.concat( [ df , df_history ] , axis=0 , join="inner" )
    df.to_csv( history_file )
else:
    df_history.to_csv( history_file )
