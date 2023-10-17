# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(r'/root/autodl-tmp/Pointcloud2023/')
# os.chdir('/root/autodl-tmp/Pointcloud2023/')
import numpy as np
import gc
from GAN2023.Z1_classes import Model, TrainModel


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# 创建Model类
reg_coeff, dro_coeff, output_num = 0.01, 0.2, 6
output_type, opt, loss = 'linear', 'adam', 'mse'
model = Model(reg_coeff, dro_coeff, output_num, output_type, opt, loss)
# 创建训练
epochs, batch_size = 100, 32
trainmodel = TrainModel(epochs, batch_size)
# 导入数据
D = np.load('GAN2023/datasets/all_data.npz')
train_in, train_out = D['train_xyz'], D['train_o']
test_in, test_out = D['test_xyz'], D['test_o']
# 保存结果
path = 'GAN2023/results/Z0/'
labels = {'3DCNN': '3DCNN', '1DCNN': '1DCNN', 'ANN': 'ANN'}


""" 3DCNN """
voxel_num, channel_num = 64, 3
vpc = D['xyz_vpc']


def voxel_match(xyz):
    xyz_new = np.zeros((xyz.shape[0], voxel_num, voxel_num, voxel_num, channel_num))
    for p in range(vpc.shape[0]):
        xyz_new[:, vpc[p, 0], vpc[p, 1], vpc[p, 2], :] = xyz[:, p, :]
    return xyz_new


train_in = voxel_match(train_in)
test_in = voxel_match(test_in)

e_m = []
a_m = []
for index in range(30):
    print(index + 1)
    # 创建3DCNN类
    cnn_3d_model = model.cnn_3d(voxel_num, channel_num)
    # print(cnn_3d_model.summary())
    # 模型训练
    trained_model, eval_metrics, accuracy_metrics = trainmodel.train_model(
        cnn_3d_model, train_in, train_out, test_in, test_out,
        path, labels['3DCNN'])

    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + labels['3DCNN'] + '/result.npz', e_m=e_m, a_m=a_m)


""" 1DCNN """
voxel_num, channel_num = 10841, 3

e_m = []
a_m = []
for index in range(30):
    print(index + 1)
    # 创建1DCNN模型
    cnn_1d_model = model.cnn_1d(voxel_num, channel_num)
    # print(cnn_1d_model.summary())
    # 模型训练
    trained_model, eval_metrics, accuracy_metrics = trainmodel.train_model(
        cnn_1d_model, train_in, train_out, test_in, test_out,
        path, label=labels['1DCNN'])

    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + labels['1DCNN'] + '/result.npz', e_m=e_m, a_m=a_m)


""" ANN """
voxel_num, channel_num = 10841, 3

e_m = []
a_m = []
for index in range(30):
    print(index + 1)
    # 创建ANN模型
    ann_model = model.ann(voxel_num, channel_num)
    # print(ann_model.summary())
    # 模型训练
    trained_model, eval_metrics, accuracy_metrics = trainmodel.train_model(
        ann_model, train_in, train_out, test_in, test_out,
        path, label=labels['ANN'])

    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + labels['ANN'] + '/result.npz', e_m=e_m, a_m=a_m)
