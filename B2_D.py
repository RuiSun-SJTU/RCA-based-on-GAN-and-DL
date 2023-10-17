# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model, losses

from GAN2023.B3_conv import ConvD1, Dense
C, D = ConvD1(), Dense()


# D模块
class Disc:
    def __init__(self):
        self.in_ = 6
        self.out_r = 10841
        self.out_c = 3

    def disc(self, loss=False):
        
        input1 = layers.Input(shape=[self.in_, ])
        input2 = layers.Input(shape=[self.out_r, self.out_c])
        
        if loss == 'dcgan' or loss == 'lsgan':
            bn = 'BN'
        elif loss == 'wgangp':
            bn = False
        else:
            bn = False
            
        down_s = [
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(128, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            C.downsample(3, 7, 3, pad='same', BN=bn, ACT='relu', MP=False),
            ]
        
        mid1 = layers.Flatten()
        
        mid2 = layers.Concatenate()

        dense_s = [
            D.densesample(256, reg=True, BN=False, ACT='relu'),
            D.densesample(256, reg=True, BN=False, ACT='relu'),
            D.densesample(256, reg=True, BN=False, ACT='relu'),
            D.densesample(256, reg=True, BN=False, ACT='relu'),
            ]
        
        if loss == 'dcgan' or loss == 'lsgan':
            last = D.densesample(256, reg=False, BN=False, ACT='classification')
        elif loss == 'wgangp':
            last = D.densesample(256, reg=False, BN=False, ACT=False)
        else:
            last = D.densesample(256, reg=False, BN=False, ACT=False)

        x = input2
        
        for down in down_s:
            x = down(x)
        
        x = mid1(x)
        
        x = mid2([x, input1])

        for dense in dense_s:
            x = dense(x)
            
        output = last(x)

        return Model(inputs=[input1, input2], outputs=output)


class Discloss:
    
    def __init__(self):
        self.LAMBDA = 0.1
        self.gp = 0.0

    def discloss(self, disc_gen_output, disc_real_output,
                 loss=None, disc=None, x=None, target=None, gen_out=None):
        
        if loss == 'dcgan':
            loss_object = losses.BinaryCrossentropy(from_logits=False)
            disc_loss = loss_object(tf.zeros_like(disc_gen_output), disc_gen_output)
            real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
            gp = self.gp
        
        elif loss == 'lsgan':
            disc_loss = 0.5 * tf.reduce_mean(tf.square(disc_gen_output))
            real_loss = 0.5 * tf.reduce_mean(tf.square(disc_real_output - 1))
            gp = self.gp
        
        elif loss == 'wgangp':
            disc_loss = tf.reduce_mean(disc_gen_output)
            real_loss = -tf.reduce_mean(disc_real_output)
            
            t = tf.random.uniform([target.shape[0], 1, 1])
            interplate = t * target + (1 - t) * gen_out
            with tf.GradientTape() as tape:
                tape.watch([interplate])
                disc_interplate_out = disc([x, interplate], training=True)
            grads = tape.gradient(disc_interplate_out, interplate)
            grad = tf.reshape(grads, [grads.shape[0], -1])
            l2_grad = tf.norm(grad, axis=1)
            gp = tf.reduce_mean(tf.square(l2_grad - 1))
        
        else:
            disc_loss = tf.reduce_mean(disc_gen_output)
            real_loss = -tf.reduce_mean(disc_real_output)
            gp = self.gp

        total_disc_loss = disc_loss + real_loss + self.LAMBDA * gp
        return total_disc_loss, disc_loss, real_loss, gp
