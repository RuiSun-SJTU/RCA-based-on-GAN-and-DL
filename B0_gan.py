# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(r'/root/autodl-tmp/Pointcloud2023/')
# os.chdir('/root/autodl-tmp/Pointcloud2023/')
import numpy as np
import tensorflow as tf
import os
import time
from IPython import display
from GAN2023.A1_data_pre import DataSample
from GAN2023.B1_G import Gen, Genloss
from GAN2023.B2_D import Disc, Discloss
from GAN2023.B4_trainview import Trainview
from GAN2023.B5_metrics import Metrics

loss, net, l1 = 'wgangp', 'yes', 'L1'
Path = './GAN2023/model/' + loss + '_' + net + '_' + l1 + '/'
os.makedirs(Path, exist_ok=True)

"""step1: 创建实例"""
datasample, draw, metrics = DataSample(), Trainview(), Metrics()
g, gloss, d, dloss = Gen(), Genloss(), Disc(), Discloss()

"""step2：构建模型"""
generator = g.gen(net)
# print(generator.summary())
discriminator = d.disc(loss)
# print(discriminator.summary())

"""step3：优化器和节点"""
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = Path + 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ackpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator, discriminator=discriminator)


# """step4：单步训练"""
@tf.function
def train_step(x, y):

    for index_1 in range(10):
        
        if index_1 < 1:
            with tf.GradientTape() as gen_tape:
                gen_out = generator(x, training=True) 
                d_gen_out = discriminator([x, gen_out], training=True)
                
                total_gen_loss, gen_loss, l1_loss = gloss.genloss(d_gen_out, gen_out, y, loss=loss, l1=l1)
            
            g_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
            
        if index_1 < 9:
            with tf.GradientTape() as disc_tape:
                gen_out = generator(x, training=True) 
                d_gen_out = discriminator([x, gen_out], training=True)
                d_real_out = discriminator([x, y], training=True)
                
                total_disc_loss, disc_loss, real_loss, gp = dloss.discloss(d_gen_out, d_real_out, loss=loss,
                                                                           disc=discriminator, x=x,
                                                                           target=y, gen_out=gen_out)
            
            d_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    return [total_gen_loss, gen_loss, l1_loss, total_disc_loss, disc_loss, real_loss, gp]


# """step5：整体训练"""
def train_fit(epochs):
   
    start = time.time()
    los, losss, maes = [], [], []
    
    for epoch, (train_x, train_y) in train_xy.repeat().take(epochs).enumerate():

        gd = train_step(train_x, train_y)
        los.append(gd)
        
        # 可视化进程
        if (epoch + 1) % 5 == 0:
            print('.', end='', flush=True)
        
        if (epoch + 1) % 25 == 0:
            los_ = np.mean(los, axis=0)
            losss.append(los_)
            if (epoch + 1) % 100 == 0:
                print('\n\nEpoch:{:d}'.format(epoch + 1))
                print('time:{:.6f}'.format(time.time()-start))
                start = time.time()
                print(f'loss_G:{los_[0]:.6f} {los_[1]:.6f} {los_[2]:.6f}')
                print(f'loss_D:{los_[3]:.6f} {los_[4]:.6f} {los_[5]:.6f} {los_[6]:.6f} \n')
            los = []
            
        # 绘制图形
        if (epoch + 1) % 100000 == 0:
            path = Path + 'training_view/'
            os.makedirs(path, exist_ok=True)
            display.clear_output(wait=True)
            nn = 88
            draw.training_plot(xyz_nominal, generator, test_x[nn:nn+1, :], test_y[nn:nn+1, :], xyz_h, xyz_c,
                               path, epoch)
        
        # 保存节点
        if (epoch + 1) % 25 == 0:
            fake = generator(test_x, training=True)
            mae = np.mean(np.abs(test_y - fake))
            maes.append(mae)
            if (epoch + 1) % 100 == 0:
                print('\nMae={:.6f}\n\n\n'.format(mae))
            
            if epoch > 3000:
                checkpoint.save(file_prefix=checkpoint_prefix)
            
    np.savetxt(Path + 'losss.csv', losss, fmt='%.5f', delimiter=',')
    np.savetxt(Path + 'Mae.csv', maes, fmt='%.5f', delimiter=',')

    return losss


# 主程序
BATCH = 32
_, xyz_nominal, train_xy, test_x, test_y, xyz_h, xyz_c = datasample.data_in(BATCH)
Losss = train_fit(epochs=5000)
