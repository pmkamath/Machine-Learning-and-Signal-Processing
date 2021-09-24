# -*- coding: utf-8 -*-
"""
file: digit_classifier.py
description: A shallow meural network for handwritten digit classification
trained on MNIST database
@author: prasad
"""

import tensorflow as tf
import numpy as np
from PIL import Image as im

filepath = "./digit-classifier-model.h5"

#Set these flags to train or load a pre-saved model and perform prediction
isTrainMode=False
isLoadandPredict=True


#load MNIST dataset
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")

#display an image form the training set, convert numpy array to image object using Pillow lib
#img= im.fromarray(x_train[0])
#img.show()


#get size of data set
#get size of each input image
xtrain_sz,m_x,n_x= x_train.shape
xtest_sz,m_xtest,n_xtest=x_test.shape
ytrain_sz= y_train.shape

x_train_reshaped= x_train.reshape(xtrain_sz, m_x*n_x)
x_test_reshaped= x_test.reshape(xtest_sz, m_xtest*n_xtest)
#input size 
input_sz=m_x*n_x

y_train_list=[]
for y in y_train:
    #create an array of zero values with size equal to 10 (0 to 9 digits)
    arr=np.zeros(10)
    arr[y]=1
    y_train_list.append(arr)
    
y_train_reshaped = np.array(y_train_list)

y_test_list=[]
for y in y_test:
    #create an array of zero values with size equal to 10 (0 to 9 digits)
    arr=np.zeros(10)
    arr[y]=1
    y_test_list.append(arr)
    
y_test_reshaped = np.array(y_test_list)

#create a model and train it  
if(isTrainMode):       
    # build model with 3 layers: 784-> 8 -> 10
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(8, input_dim=input_sz, activation="softmax"),
      tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    # choose optimiser
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)
    
    # compile model
    model.compile(optimizer=optimizer, loss='mse')
    
    # train model
    model.fit(x_train_reshaped, y_train_reshaped, epochs=20)
    
    # evaluate model on test set (cross validation using hold out)
    print("\nEvaluation on the test set:")
    model.evaluate(x_test_reshaped,  y_test_reshaped, verbose=2)
    
    #save the trained model to a HD5 file
    tf.keras.models.save_model(
        model,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )

#load a trained model and perform digit prediction
elif (isLoadandPredict):
    customTest=True
    customTestOutput = 2
    #load a saved model
    loaded_model = tf.keras.models.load_model(filepath)
    if (not customTest):
        # get predictions
        data_idx =789
        data = x_test_reshaped[data_idx]
        data = np.array([data])
        #display an image form the training set, convert numpy array to image object using Pillow lib
        img= im.fromarray(x_test[data_idx])
        img.show()
        print("The expected value is: {}".format(y_test[data_idx]))
    else:
        img = im.open("test.png")
        img= img.convert("1")
        img_arr=np.array(img)
        shape_img=img_arr.shape
        data= img_arr.reshape(shape_img[0]*shape_img[1])
        data = np.array([data])
        #img.show()
    predictions = loaded_model.predict(data)
    
    print("The expected value is: {}".format(customTestOutput))
    
    #find max probability
    max_idx=0
    max_val=predictions[0][0]
    for idx in range(1,len(predictions[0])):
        if(predictions[0][idx]>max_val):
            max_idx=idx
            max_val=predictions[0][idx]
    
    print("The predicted value is: {}".format(max_idx))
    

