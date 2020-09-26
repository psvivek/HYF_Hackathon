import datetime

from django.db import models
from django.utils import timezone

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import glob 
import os 
import cv2
import math
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Convolution2D,Activation,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from doc_loader import DocumentLoader, OutputType

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import from doc_loader package doc_loader module
# from doc_loader import doc_loader
# Import Filestorage from werkzeug library to use FileStorage data structure
from werkzeug.datastructures import FileStorage
# Using Image from PILLOW library
from PIL import Image

def load_model(weights_path, img_width = 200, img_height = 200):    
    """
    Loads the model given the weights path

    Arguments: 
        weights_path [str] -- Path where weights will be stored
        img_width [str] -- Width of the image 
        img_height [str] -- Height of the image
    Returns:
        model [keras model object] -- Keras model with weights loaded
    """
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(3,3), padding='same', input_shape = (img_width,img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3)))

    model.add(Conv2D(16, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])    
    
    model.load_weights(weights_path)
    return model

def infer_skin_model(file_path, model,  plot = False):
  
    
    img_width, img_height =  200, 200
    file_path = "./polls"+file_path
    page_count, img_array_lst = DocumentLoader.load(file_path, max_num_pages = 1, output_type=OutputType.NUMPY)
#     Plotting raw images and cropped sign boxes
    if plot:
        plt.rcParams["figure.figsize"] = (20,15)
        plt.imshow(img_array_lst[0])
        plt.title("Skin Pic")
        plt.show()
        
#     Preprocessing steps like resezing and scaling the pixel
    img_arr = cv2.resize(img_array_lst[0], (img_width, img_height)).astype('float32')/255
    img_arr = img_arr.reshape((-1, img_width, img_height, 3))
        
    result_dict = {}
    print("img_arr = ", model.predict(img_arr))


# Predicting and creation of result dictionary    
    if np.argmax(model.predict(img_arr), axis = -1)[0] == 0:
        result_dict['disease_class'] = 'acne_skin'
        result_dict['confidence'] = np.max(model.predict(img_arr))
    elif np.argmax(model.predict(img_arr), axis = -1)[0] == 1:
        result_dict['disease_class'] = 'normal_skin'
        result_dict['confidence'] = np.max(model.predict(img_arr))        
    elif np.argmax(model.predict(img_arr), axis = -1)[0] == 2:
        result_dict['disease_class'] = 'psoriasis_skin'
        result_dict['confidence'] = np.max(model.predict(img_arr))         
          
    return result_dict
    
def infer_dental_model(file_path, model,  plot = False):
  
    
    img_width, img_height =  200, 200
    file_path = "./polls"+file_path
    page_count, img_array_lst = DocumentLoader.load(file_path, max_num_pages = 1, output_type=OutputType.NUMPY)
#     Plotting raw images and cropped sign boxes
    if plot:
        plt.rcParams["figure.figsize"] = (20,15)
        plt.imshow(img_array_lst[0])
        plt.title("Mouth Pic")
        plt.show()
        
#     Preprocessing steps like resezing and scaling the pixel
    img_arr = cv2.resize(img_array_lst[0], (img_width, img_height)).astype('float32')/255
    img_arr = img_arr.reshape((-1, img_width, img_height, 3))
        
    result_dict = {}
#     print("img_arr = ", model.predict(img_arr))

# Predicting and creation of result dictionary    
    if np.argmax(model.predict(img_arr), axis = -1)[0] == 0:
        result_dict['disease_class'] = 'normal_teeth'
        result_dict['confidence'] = np.max(model.predict(img_arr))
    elif np.argmax(model.predict(img_arr), axis = -1)[0] == 1:
        result_dict['disease_class'] = 'dental_caries'
        result_dict['confidence'] = np.max(model.predict(img_arr))        
    elif np.argmax(model.predict(img_arr), axis = -1)[0] == 2:
        result_dict['disease_class'] = 'periodontitis'
        result_dict['confidence'] = np.max(model.predict(img_arr))         
    print(result_dict)
    return result_dict
    
skin_model = load_model(weights_path = "./polls/static/skin_model_weights.h5", img_width = 200, img_height = 200)
dental_model = load_model(weights_path = "./polls/static/dental_model_weights.h5", img_width = 200, img_height = 200)

# img_file_path = "image.jpeg"

def dentalPreds(img_file_path):
    # print(img_file_path)
    return infer_dental_model(file_path = img_file_path, model = dental_model,  plot = False)

def skinPreds(img_file_path):
    return infer_skin_model(file_path = img_file_path, model = skin_model,  plot = False)   

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now
    was_published_recently.admin_order_field = 'pub_date'
    was_published_recently.boolean = True
    was_published_recently.short_description = 'Published recently?'

class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text
