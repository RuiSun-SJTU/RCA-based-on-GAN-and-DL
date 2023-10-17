# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, regularizers


# 一维卷积
class ConvD1:

    # 上采样
    @staticmethod
    def upsample(filters, size, stride, pad=False, BN=False, ACT=False, MP=False):
        
        initializer = tf.random_normal_initializer(0., 0.02) 
        upmodel = tf.keras.Sequential() 

        if pad == 'same':
            upmodel.add(layers.Conv1DTranspose(filters, size, stride, padding='same', 
                                               kernel_initializer=initializer, use_bias=False)) 
        elif pad == 'valid':
            upmodel.add(layers.Conv1DTranspose(filters, size, stride, 
                                               kernel_initializer=initializer, use_bias=False)) 
        
        if BN == 'BN':
            upmodel.add(layers.BatchNormalization()) 
        elif BN == 'LN':
            upmodel.add(layers.LayerNormalization())
            
        if ACT == 'relu':
            upmodel.add(layers.LeakyReLU())
        elif ACT == 'tanh':
            upmodel.add(layers.Activation('tanh'))
        elif ACT == 'classification':
            upmodel.add(layers.Activation('sigmoid'))
        elif ACT == 'classifications':
            upmodel.add(layers.Softmax())
            
        if MP:
            upmodel.add(layers.MaxPooling1D())
        
        return upmodel
    
    # g 下采样
    @staticmethod
    def downsample(filters, size, stride, pad=False, BN=False, ACT=False, MP=False):
        
        initializer = tf.random_normal_initializer(0., 0.02) 
        downmodel = tf.keras.Sequential() 
        
        if pad == 'same':
            downmodel.add(layers.Conv1D(filters, size, stride, padding='same', 
                                        kernel_initializer=initializer, use_bias=False)) 
        elif pad == 'valid':
            downmodel.add(layers.Conv1D(filters, size, stride, 
                                        kernel_initializer=initializer, use_bias=False)) 
        
        if BN == 'BN':
            downmodel.add(layers.BatchNormalization()) 
        elif BN == 'LN':
            downmodel.add(layers.LayerNormalization())
        
        if ACT == 'relu':
            downmodel.add(layers.LeakyReLU())
        elif ACT == 'tanh':
            downmodel.add(layers.Activation('tanh'))
        elif ACT == 'classification':
            downmodel.add(layers.Activation('sigmoid'))
        elif ACT == 'classifications':
            downmodel.add(layers.Softmax())
            
        if MP:
            downmodel.add(layers.MaxPooling1D())
        
        return downmodel


# 全连接
class Dense:

    @staticmethod
    def densesample(units, reg=False, BN=False, ACT=False):
        
        initializer = tf.random_normal_initializer(0., 0.02) 
        densemodel = tf.keras.Sequential()
        
        if reg:
            densemodel.add(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01),
                                        kernel_initializer=initializer, use_bias=True))
        else:
            densemodel.add(layers.Dense(units,
                                        kernel_initializer=initializer, use_bias=True))
        
        if BN == 'BN':
            densemodel.add(layers.BatchNormalization()) 
        elif BN == 'LN':
            densemodel.add(layers.LayerNormalization())
        
        if ACT == 'relu':
            densemodel.add(layers.LeakyReLU())
        elif ACT == 'tanh':
            densemodel.add(layers.Activation('tanh'))
        elif ACT == 'classification':
            densemodel.add(layers.Activation('sigmoid'))
        elif ACT == 'classifications':
            densemodel.add(layers.Softmax())
        
        return densemodel
