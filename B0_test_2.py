# 不同模型生成点云图像绘制
import tensorflow as tf
from GAN2023.A1_data_pre import DataSample
from GAN2023.B1_G import Gen
from GAN2023.B4_trainview import Trainview
datasample, draw, g = DataSample(), Trainview(), Gen()

BATCH = 32
_, xyz_nominal, _, train_x, train_y, test_x, test_y, xyz_h, xyz_c = datasample.data_in_(BATCH)

loss, net, l1 = 'wgangp', 'yes', 'L1'
Path = './GAN2023/model/' + loss + '_' + net + '_' + l1 + '/'
generator = g.gen(net)
checkpoint_dir = Path + 'training_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

A = 'GAN2023/results/Z3/'
B = tf.constant(0)
nn = 489
draw.training_plot(xyz_nominal, generator, train_x[nn:nn + 1, :], train_y[nn:nn + 1, :],
                   xyz_h, xyz_c)

# A = 'GAN2023/results/Z2/'
# B = tf.constant(0)
# nn = 68
# draw.training_plot(xyz_nominal, generator, test_x[nn:nn + 1, :], test_y[nn:nn + 1, :],
#                    xyz_h, xyz_c, A, B)
