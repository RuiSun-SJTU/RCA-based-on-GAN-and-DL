# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model, losses

from GAN2023.B3_conv import ConvD1, Dense
C, D = ConvD1(), Dense()


# G模块
class Gen:
    def __init__(self):
        self.in_ = 6

    def gen(self, net=False):
        
        inputs = layers.Input(shape=[self.in_, ])
        
        first_s = [
            D.densesample(10841*3, reg=True, BN=False, ACT='relu'),   # 10841*3
            ]
        
        mid = layers.Reshape((10841, 3))

        down_s = [
            C.downsample(256, 6, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 3612
            C.downsample(256, 4, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 1203
            C.downsample(256, 5, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 400
            C.downsample(256, 5, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 132
            C.downsample(256, 4, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 43
            C.downsample(512, 5, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 13
            C.downsample(512, 5, 3, pad='valid', BN=False, ACT='relu', MP=False),   # 3
            ]
        
        up_s = [
            C.upsample(512, 7, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 13
            C.upsample(256, 7, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 43
            C.upsample(256, 6, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 132
            C.upsample(256, 7, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 400
            C.upsample(256, 6, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 1203
            C.upsample(256, 6, 3, pad='valid', BN=False, ACT='relu', MP=False),     # 3612
            ]
        
        last_s = [
            C.upsample(3, 8, 3, pad='valid', BN=False, ACT='tanh', MP=False)        # 10841
            ]

        x = inputs
        
        for first in first_s:
            x = first(x)
            
        x = mid(x)
        
        if net == 'yes':
            skips = []
            for down in down_s:
                x = down(x)
                skips.append(x)
                
            skips = reversed(skips[:-1])
            for up, skip in zip(up_s, skips):
                x = up(x)
                x = layers.Concatenate()([x, skip])
                
        elif net == 'no':
            for down in down_s:
                x = down(x)
                
            for up in up_s:
                x = up(x)
                                                                
        for last in last_s:
            x = last(x)

        return Model(inputs=inputs, outputs=x)


class Genloss:
    def __init__(self):
        # self.Lambda = 0.01
        self.Lambda = 0.1

    def genloss(self, disc_gen_output, gen_output, target, loss=False, l1=False):
        
        if loss == 'dcgan':
            loss_object = losses.BinaryCrossentropy(from_logits=False)
            gen_loss = loss_object(tf.ones_like(disc_gen_output), disc_gen_output)
        elif loss == 'lsgan':
            gen_loss = 0.5 * tf.reduce_mean(tf.square(disc_gen_output - 1))
        elif loss == 'wgangp':
            gen_loss = -tf.reduce_mean(disc_gen_output)
        else:
            gen_loss = -tf.reduce_mean(disc_gen_output)

        if l1 == 'MAE':
            l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
        elif l1 == 'MSE':
            l1_loss = tf.reduce_mean(tf.square(gen_output - target))
        elif l1 == 'L1':
            l1_loss = tf.norm(target - gen_output, ord=1)
        elif l1 == 'L2':
            l1_loss = tf.norm(target - gen_output, ord=2)
        else:
            l1_loss = 0.0

        total_gen_loss = gen_loss + self.Lambda * l1_loss
        return total_gen_loss, gen_loss, l1_loss
