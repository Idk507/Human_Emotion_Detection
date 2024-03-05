import tensorflow as tf 
import numpy as np 
import cv2 
import sklearn 
import io 
import os 
import time  
import random 
import PIL
from PIL import Image 
from sklearn.metrics import confusion_matrix,roc_curve
import matplotlib.pyplot as plt
import datetime 
import pathlib 
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow_probability as tfds 
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy,BinaryCrossentropy,CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, VGG19, InceptionV3, Xception, InceptionResNetV2 
from tensorflow.keras.regularizers import L1,L2
import configuration


resize_rescale_layers = tf.keras.Sequential([
    Resizing(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"]),
    Rescaling(1./255)
])


lenet_model = tf.keras.Sequential([
    InputLayer(input_shape=(None,None,3)),
    resize_rescale_layers,
    Conv2D(filters=CONFIGURATION['N_FILTERS'],kernel_size=CONFIGURATION['KERNEL_SIZE'],strides=CONFIGURATION['N_STRIDES'],padding='valid',activation='relu',kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION['POOL_SIZE'],strides=CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate=CONFIGURATION['DROPOUT_RATE']),
    
    Conv2D(filters=CONFIGURATION['N_FILTERS']*2 +4,kernel_size=CONFIGURATION["KERNEL_SIZE"],strides=CONFIGURATION["N_STRIDES"],padding='valid',activation='relu',kernel_regularizer = L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION["POOL_SIZE"],strides=CONFIGURATION["N_STRIDES"]*2),
    Flatten(),
    
    Dense(CONFIGURATION["N_DENSE_1"],activation='relu',kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate=CONFIGURATION["DROPOUT_RATE"]),
    
    Dense(CONFIGURATION["N_DENSE_2"],activation='relu',kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    
    Dense(CONFIGURATION["NUM_CLASSES"],activation='softmax')
])

loss_function = CategoricalCrossentropy(from_logits=False)

lenet_model.compile(optimizer=Adam(learning_rate=CONFIGURATION['LEARNING_RATE']),loss=loss_function,metrics=['accuracy'])


history = lenet_model.fit(train_dataset, validation_data=validation_dataset, epochs=5, verbose=2)

lenet_model.evaluate(validation_dataset)

test_image = cv2.imread(r"C:\Users\danus\Downloads\human_emotion_detection\archive\EmotionsDataset\data\angry\0.jpg")
im = tf.constant(test_image,dtype=tf.float32)
print(im.shape)
im = tf.expand_dims(im,axis=0)
v = tf.argmax(lenet_model(im),axis=-1).numpy()[0]

CONFIGURATION["CLASS_NAMES"][v]

lenet_model.save('lenet_model')

lenet_model.save_weights("lenet_model_weights.h5")
