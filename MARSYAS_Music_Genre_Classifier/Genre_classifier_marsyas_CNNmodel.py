# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 00:00:46 2021

@author: prasad
"""

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
    data_x=np.array(x)
    data_y =data["labels"]
    data_y =np.array(data_y)
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.25)
    x_train,x_validation,y_train,y_validation  = train_test_split(x_train,y_train, test_size=0.2)
    # add an axis to input sets
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return [ x_train, x_test, y_train, y_test , x_validation, y_validation]
    
x_train, x_test, y_train, y_test, x_validation, y_validation=load_marsyas_data("marsyas_mfcc_dat.json")
model = keras.models.Sequential()

input_shape = (x_train.shape[1], x_train.shape[2], 1)

# 1st conv layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# 2nd conv layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# 3rd conv layer
model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# flatten output and feed it into dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))


# output layer
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
# choose optimiser
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# compile model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=50, batch_size=32,shuffle=True)

# evaluate model on test set
print("\nEvaluation on the test set:")
model.evaluate(x_test,  y_test, verbose=2)
