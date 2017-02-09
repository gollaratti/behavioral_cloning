# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:22:03 2017

References:
    [1] Comma ai model: https://github.com/commaai/research/blob/master/train_steering_model.py
    [2] Data augmentation methods: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.7vrqkgnvl
    [3] NVIDIA deep learning paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

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

#camera image
ch_raw, row_raw, col_raw = 3, 160, 320

##################################################
#model architecture - inspired by reference [3] 
##################################################
def get_model():
  row, col, ch = 20, 80, 3  
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
  model.add(Convolution2D(8, 5, 5, border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.50))
  model.add(Convolution2D(16, 5, 5, border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.50))  
  model.add(Convolution2D(32, 5, 5, border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.50))  
  model.add(Convolution2D(32, 3, 3, border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.50))  
  model.add(Convolution2D(64, 3, 3, border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.50))  
  model.add(Flatten())
  model.add(Dense(64))
  model.add(Dropout(.50))
  model.add(Dense(32))
  model.add(Dropout(.50))
  model.add(Dense(16))
  model.add(Dropout(.50))  
  model.add(Dense(1))
  model.add(ELU())
  
  
  adam_grad = Adam(lr=0.0001)
  model.compile(optimizer=adam_grad, loss="mean_squared_error")

  return model   

#######################################################  
# pre-processing and data augmentation reference: [2]
#######################################################
def preprocess(image):
    #cut out the bottom and upper portion of the image
    image_pr = image[60:(row_raw-20), 0:col_raw]
    #resize to row:column - 20:80 (1/4th the cut out portion)
    image_pr = cv2.resize(image_pr, (80,20), interpolation=cv2.INTER_AREA)
    return image_pr
    

def trans_image(image,steer,trans_range):
    # Translation - shift the image
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(col_raw,row_raw))
    return image_tr,steer_ang

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image    
    
#######################################################
# image data processing
#######################################################
file_list = listdir('./data/IMG')
num_images = (24108*12)
print('number of images including augmented images:',num_images)

image_data = np.ndarray((num_images,20,80,3))
steering_angle = np.ndarray((num_images),dtype=float)
print("reading image data..")
# iterate throught the images
i=0
# temp variable for saving images 
j=-2
with open('./data/driving_log.csv', 'r') as csvfile:
    freader = csv.reader(csvfile)
    for row in freader:
        j+=1
        if(row[0] != 'center' and i < num_images):
            ##################################################
            # center data
            #################################################
            center_raw_image = mpimg.imread('./data/'+row[0])
            #center_raw_image = cv2.cvtColor(center_raw_image,cv2.COLOR_BGR2RGB)
            image_data[i] = preprocess(center_raw_image)
            steering_angle[i] = float(row[3])
            i+=1
            if(j==1):
                plt.imsave('center_raw_image.jpg',np.uint8(center_raw_image))
                plt.imsave('center_raw_image_preprocessed.jpg',np.uint8(image_data[i-1]))               
            
            # center data random shadow
            image_data[i] = preprocess(add_random_shadow(center_raw_image))
            steering_angle[i] = steering_angle[i-1]
            i+=1
            if(j==1):
                plt.imsave('center_random_shadow_image.jpg',np.uint8(image_data[i-1]))  
            
            #flip data
            image_data[i] = preprocess(cv2.flip(center_raw_image,1))
            steering_angle[i] = -1.0*float(steering_angle[i-1])
            i+=1
            if(j==1):
                plt.imsave('center_flip.jpg',np.uint8(image_data[i-1]))  
            
            #horizontal and vertical shift data
            raw_image_trans, steering_angle[i] = trans_image(center_raw_image,float(row[3]),100)
            image_data[i] = preprocess(raw_image_trans)
            i+=1
            if(j==1):
                plt.imsave('center_shift_data_100.jpg',np.uint8(image_data[i-1]))   
            
            #horizontal and vertical shift data
            raw_image_trans, steering_angle[i] = trans_image(center_raw_image,float(row[3]),40)
            image_data[i] = preprocess(raw_image_trans)
            i+=1
            if(j==1):
                plt.imsave('center_shift_data_40.jpg',np.uint8(image_data[i-1])) 
            
            #centre brightness
            image_data[i] = preprocess(augment_brightness_camera_images(center_raw_image))
            steering_angle[i] = float(row[3])
            i+=1
            if(j==1):
                plt.imsave('center_brightness.jpg',np.uint8(image_data[i-1])) 
            
            ##################################################
            # left data
            #################################################      
            #brightness right data
            left_raw_image = mpimg.imread('./data/'+row[1])
            #left_raw_image = cv2.cvtColor(left_raw_image,cv2.COLOR_BGR2RGB)
            image_data[i] = preprocess(augment_brightness_camera_images(left_raw_image))         
            steering_angle[i] = float(row[3])+0.25
            i+=1
            if(j==1):
                plt.imsave('left_brightness_image.jpg',np.uint8(image_data[i-1]))
            
            #left data
            image_data[i] = preprocess(left_raw_image)
            steering_angle[i] = float(row[3])+0.25
            i+=1
            if(j==1):
                plt.imsave('left_raw_image.jpg',np.uint8(left_raw_image))
                plt.imsave('left_raw_image_preprocessed.jpg',np.uint8(image_data[i-1])) 
   

            # left  random shadow
            if(i%2 == 0):            
                image_data[i] = preprocess(add_random_shadow(left_raw_image))
                steering_angle[i] = float(row[3])+0.25
                i+=1
            if(j==1):
                plt.imsave('left_shadow.jpg',np.uint8(image_data[i-1]))              
            
            #flip data
            if(i%2 == 1):
                raw_image_flip = cv2.flip(left_raw_image,1)
                image_data[i] = preprocess(raw_image_flip)
                steering_angle[i] = -1.0*(float(row[3])+0.25)
                i+=1
            if(j==1):
                plt.imsave('left_flip_processed.jpg',np.uint8(image_data[i-1])) 
                
            ##################################################
            # Right data
            #################################################             
            #brightness right data
            right_raw_image = mpimg.imread('./data/'+row[2])
            #right_raw_image = cv2.cvtColor(right_raw_image,cv2.COLOR_BGR2RGB)
            image_data[i] = preprocess(augment_brightness_camera_images(right_raw_image))       
            steering_angle[i] = float(row[3])-0.25
            i+=1
            if(j==1):
                plt.imsave('right_brightness.jpg',np.uint8(image_data[i-1]))            
                
            #right data        
            image_data[i] = preprocess(right_raw_image)
            steering_angle[i] = float(row[3])-0.25
            i+=1
            if(j==1):
                plt.imsave('right_raw_image.jpg',np.uint8(right_raw_image))
                plt.imsave('right_raw_image_preprocessed.jpg',np.uint8(image_data[i-1]))    
  
            # right    random shadow
            if(i%2 == 1):
                image_data[i] = preprocess(add_random_shadow(right_raw_image))
                steering_angle[i] = float(row[3])-0.25
                i+=1
            if(j==1):
                plt.imsave('right_random_shadow.jpg',np.uint8(image_data[i-1]))             
            
            #flip data
            if(i%2 == 0):            
                image_data[i] = preprocess(cv2.flip(right_raw_image,1))
                steering_angle[i] = -1.0*(float(row[3])-0.25)
                i+=1
            if(j==1):
                plt.imsave('right_flip.jpg',np.uint8(image_data[i-1]))                      
  

#######################################################
# model training and validation
#######################################################
print("creating model")
model_steering = get_model()
model_steering.summary()


print("training and validating the steering model")
history = model_steering.fit(image_data, steering_angle, batch_size=64, nb_epoch=5, verbose=1, validation_split=0.1, shuffle=True)

print("Saving steering model and weights")
json_string = model_steering.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_string)   
model_steering.save_weights("model.h5")
print("Saved steering model to disk")