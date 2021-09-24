# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:18:21 2021

@author: prasad
"""
import json
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

#import the json data 

def load_marsyas_data(json_path):
    openfile=open(json_path, "r")
    data= json.load(openfile)
    x=data['mfcc']
    # the mfcc vectors for the default configuration 
    # is a 3D array ( 50 * 259 * 13), there are 50 segments:
    # 5 segments of one type of label each. each segment was of 22050*30/5
    # samples = 132300, with a hop size of 512 for STFT, we get 259 frames
    # for each frame we extract 13 MFCC coefficients
    x=np.array(x)
    #flattening the 3D array
    #x=np.reshape(x, [x.shape[0]*x.shape[1]*x.shape[2],1])
    #print(x.shape)
    data_x=np.reshape(x, [x.shape[0], x.shape[1]*x.shape[2]])
    data_y =data["labels"]
    data_y =np.array(data_y)
    '''data_y_op=[]

    for label in data_y:
        temp=[]
        for i in range(0,10):
            temp.append(0)
        temp[label]=1;
        data_y_op.append(temp)
    data_y =np.array(data_y_op)'''
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.25)
    x_train,x_validation,y_train,y_validation  = train_test_split(x_train,y_train, test_size=0.2)
    return [ x_train, x_test, y_train, y_test , x_validation, y_validation]
    
x_train, x_test, y_train, y_test, x_validation, y_validation=load_marsyas_data("marsyas_mfcc_dat.json")
model = keras.models.Sequential([
  keras.layers.Dense(512, input_dim=x_train.shape[1], activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
  keras.layers.Dropout(0.3),
  keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
  keras.layers.Dropout(0.3),
  keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
  keras.layers.Dropout(0.3),
  keras.layers.Dense(10, activation="softmax")
])

model.summary()
# choose optimiser
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# compile model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=50, batch_size=32)

# evaluate model on test set
print("\nEvaluation on the test set:")
model.evaluate(x_test,  y_test, verbose=2)
