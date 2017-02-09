# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:22:03 2017
Used for training with additional training data
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import cv2
import csv
import matplotlib.image as mpimg
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from keras.models import model_from_json

with open('model.json', 'r') as jfile:
    model = model_from_json(jfile.read())
    model.compile("adam", "mse")
    model.load_weights('model.h5')
    print("training and validating the model")
    image_data = np.load('image_da.npy')
    steering_data = np.load('steering_da.npy')
    print(image_data.shape)
    print(steering_data.shape)
    print(steering_data)
    for i in range(image_data.shape[0]):
        image_data[i] = cv2.cvtColor(image_data[i],cv2.COLOR_BGR2RGB)
    plt.imsave('image0.jpg',np.uint8(image_data[0]))
    history = model.fit(image_data, steering_data, batch_size=64, nb_epoch=5, verbose=1, validation_split=0.1, shuffle=True)
    print("Saving model and updated weights")
    json_string = model.to_json()
    with open("model_new.json", "w") as json_file:
        json_file.write(json_string)
        
    model.save_weights("model_new.h5")
    print("Saved model to disk")




