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

loss_function = CategoricalCrossentropy(from_logits=False)

