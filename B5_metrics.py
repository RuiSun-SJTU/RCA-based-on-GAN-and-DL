# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# 指标
class Metrics:

    @staticmethod
    def psnr_np(real, fake):
        mse = np.mean(np.square(real - fake))
        psnr = 10 * np.log10(2 * 2 / mse)
        return psnr

    @staticmethod
    def ssim_np(real, fake):
        a = real + 1.0
        b = fake + 1.0
        
        c1 = 2 * np.mean(a) * np.mean(b) + np.square(0.01 * 2)
        c2 = 2 * np.mean((a - np.mean(a)) * (b - np.mean(b))) + np.square(0.03 * 2)
        c3 = np.square(np.mean(a)) + np.square(np.mean(b)) + np.square(0.01 * 2)
        c4 = (np.mean(np.square(a - np.mean(a))) + 
              np.mean(np.square(b - np.mean(b))) + np.square(0.03 * 2))
        c = (c1 * c2) / (c3 * c4)
        return c

    @staticmethod
    def nrmse_np(real, fake):
        d = np.sqrt(np.sum(np.square(real - fake)) / np.sum(np.square(real)))
        return d

    @staticmethod
    def mae_np(real, fake):
        return np.mean(np.abs(real - fake))

    @staticmethod
    def psnr(real, fake):
        return tf.image.psnr(real, fake, max_val=2).numpy()

    @staticmethod
    def ssim(real, fake):
        a = (real + 1.0).astype('float32')
        b = (fake + 1.0).astype('float32')
        
        c1 = 2 * tf.reduce_mean(a) * tf.reduce_mean(b) + tf.square(0.01 * 2)
        c2 = 2 * tf.reduce_mean((a - tf.reduce_mean(a)) * (b - tf.reduce_mean(b))) + tf.square(0.03 * 2)
        c3 = tf.square(tf.reduce_mean(a)) + tf.square(tf.reduce_mean(b)) + tf.square(0.01 * 2)
        c4 = (tf.reduce_mean(tf.square(a - tf.reduce_mean(a))) + 
              tf.reduce_mean(tf.square(b - tf.reduce_mean(b))) + tf.square(0.03 * 2))
        c = (c1 * c2) / (c3 * c4)
        return c.numpy()

    @staticmethod
    def nrmse(real, fake):
        d = tf.sqrt(tf.reduce_sum(tf.square(real - fake)) / tf.reduce_sum(tf.square(real)))
        return d.numpy()
