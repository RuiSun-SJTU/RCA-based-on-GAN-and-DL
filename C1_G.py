# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from GAN2023.B1_G import Gen


class G:
    def __init__(self):
        self.d1 = 0.96
        self.d2 = 0.96
        self.d3 = -1.02
        self.d4 = -1.50

    def gen(self, x1):
        c1, c2 = self.d1, self.d2
        c3 = np.where((abs(x1[:, 0]) < c1) & (abs(x1[:, 1]) < c1) & (abs(x1[:, 2]) < c1) &
                      (x1[:, 3] < self.d3) & (x1[:, 3] > self.d4) &
                      (abs(x1[:, 4]) < c2) & (abs(x1[:, 5]) < c2))[0]

        c4 = np.random.choice(c3, 500, replace=False)

        return x1[c4, :]

    def gen1(self):
        d = np.load('GAN2023/datasets/all_data.npz')
        x0 = d['train_o'][475:].copy()
        # 故障样本加噪声
        x1 = np.tile(x0, (100, 1)) + (np.random.rand(2500, 6) - 0.5) / 10

        xx = self.gen(x1)

        return xx

    def gen2(self):
        x0 = np.random.rand(50000, 3) * 2 - 1
        x00 = np.random.rand(50000, 3) * 4 - 2
        x1 = np.concatenate((x0, x00), axis=1)

        xx = self.gen(x1)

        return xx


""" 构造原始数据、生成测试数据添加、真实测试数据添加 """
if __name__ == '__main__':
    g = G()
    x = g.gen2()

    gen = Gen()
    generator = gen.gen(net='yes')
    Path = './GAN2023/model/wgangp_yes_L1/training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(Path))

    y_f = []
    for index in range(x.shape[0]):
        fake = generator(x[index: index + 1, :], training=True).numpy()[0, :]
        y_f.append(fake)
    y_fake = np.asarray(y_f)

    np.savez('GAN2023/datasets/random_0.npz', random=x)
    np.savez('GAN2023/datasets/random_0_gan.npz', gan=y_fake)
