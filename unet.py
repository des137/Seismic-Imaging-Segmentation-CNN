"""
Module containing unet model.

"""

from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

def unet_model(pretrained_weights=None, input_size=(101,101,1)):
    
    initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    
    inputs = Input(input_size)
    input_padded = ZeroPadding2D(padding=((14, 13), (14, 13)))(inputs)  
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(input_padded)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    conv1 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    conv1 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    conv2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    conv2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv2)
    
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv3)
    conv3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv3)
    conv3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv3)
    
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv4)
    conv4 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv4)
    conv4 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv4)#(drop4)
    
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv5)
    conv5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv5)
    conv5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv5)
    
    conv5b = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool5)
    conv5b = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv5b)
    conv5b = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv5b)
    conv5b = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv5b)
    up5b = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv5b))
    
    merge6 = merge([conv5,up5b], mode='concat', concat_axis = 3)
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(merge6)
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv5))
    
    merge6 = merge([conv4,up6], mode='concat', concat_axis = 3)
    conv6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(merge6)
    conv6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    conv6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    conv6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv6)
    
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size=(2,2))(conv6))
    merge7 = merge([conv3,up7], mode='concat', concat_axis = 3)
    conv7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(merge7)
    conv7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv7)
    conv7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv7)
    conv7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv7)
    
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(merge8)
    conv8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv8)
    conv8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv8)
    conv8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv8)
    
    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(merge9)
    conv9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv9)
    conv9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv9)
    conv9 = Conv2D(2, 2, activation='relu', padding='same', kernel_initializer=initializer)(conv9)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    crop = Cropping2D(cropping=((14, 13), (14, 13)))(conv10)
    model = Model(inputs=inputs, output=crop)
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model
