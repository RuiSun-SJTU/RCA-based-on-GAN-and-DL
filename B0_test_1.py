# 不同模型生成点云质量 psnr、ssim、nrmse、mae
import numpy as np
import tensorflow as tf
from GAN2023.A1_data_pre import DataSample
from GAN2023.B1_G import Gen
from GAN2023.B5_metrics import Metrics
datasample, g, metrics = DataSample(), Gen(), Metrics()


BATCH = 32
_, _, _, train_x, train_y, test_x, test_y, _, _ = datasample.data_in_(BATCH)

loss, net, l1 = 'lsgan', 'no', 'L1'
Path = './GAN2023/model/' + loss + '_' + net + '_' + l1 + '/'
generator = g.gen(net)
checkpoint_dir = Path + 'training_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

psnr, ssim, nrmse, mae = [], [], [], []
for index in range(475, 500):
    fake = generator(train_x[index: index + 1, :], training=True)
    real = train_y[index: index + 1, :]
    psnr.append(metrics.psnr_np(real, fake))
    ssim.append(metrics.ssim_np(real, fake))
    nrmse.append(metrics.nrmse_np(real, fake))
    mae.append(metrics.mae_np(real, fake))

psnr, ssim, nrmse, mae = [], [], [], []
for index in range(test_x.shape[0]):
    fake = generator(test_x[index: index + 1, :], training=True)
    real = test_y[index: index + 1, :]
    psnr.append(metrics.psnr_np(real, fake))
    ssim.append(metrics.ssim_np(real, fake))
    nrmse.append(metrics.nrmse_np(real, fake))
    mae.append(metrics.mae_np(real, fake))

print(np.mean(psnr))
print(np.mean(ssim))
print(np.mean(nrmse))
print(np.mean(mae))

print(np.mean(psnr[50:]))
print(np.mean(ssim[50:]))
print(np.mean(nrmse[50:]))
print(np.mean(mae[50:]))
b = mae[50:]

np.savetxt(Path + 'psnr.csv', psnr, fmt='%.8f', delimiter=',')
np.savetxt(Path + 'ssim.csv', ssim, fmt='%.8f', delimiter=',')
np.savetxt(Path + 'nrmse.csv', nrmse, fmt='%.8f', delimiter=',')
np.savetxt(Path + 'mae_.csv', mae, fmt='%.8f', delimiter=',')
