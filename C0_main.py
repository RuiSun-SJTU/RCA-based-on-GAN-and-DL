# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(r'/root/autodl-tmp/Pointcloud2023/')
# os.chdir('/root/autodl-tmp/Pointcloud2023/')
import numpy as np
import gc
from GAN2023.Z1_classes import Model, TrainModel


""" part1 创建模型和训练的类 """
# 创建Model类
reg_coeff, dro_coeff, output_num = 0.01, 0.2, 6
output_type, opt, loss = 'linear', 'adam', 'mse'
model = Model(reg_coeff, dro_coeff, output_num, output_type, opt, loss)
# 创建训练
epochs, batch_size = 100, 32
trainmodel = TrainModel(epochs, batch_size)

""" part2 导入真实数据、生成数据 """
# 导入真实数据
D = np.load('GAN2023/datasets/all_data.npz')
train_x, train_y = D['train_o'], D['train_xyz']
test_x, test_y = D['test_o'], D['test_xyz']
# 导入生成数据
T = np.load('GAN2023/datasets/random_0.npz')
gan_x = T['random']
TT = np.load('GAN2023/datasets/random_0_gan.npz')
gan_y = TT['gan']

""" part3 确定样本比例 """
# 正样本比例
ratio_yes = np.linspace(0.95, 0.50, 10)
# 负样本添加数量
ratio_add = np.ceil(475 / ratio_yes - 500).astype(int)
# 负样本比例
ratio_no = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50']

# ###################################################第 num 次试验
num = 0
path = 'GAN2023/results/Z1/'
# ###################################################第 num 次试验

""" part4 训练集、测试集、验证集体素化 """
label = '3DCNN_' + ratio_no[num]
voxel_num, channel_num = 64, 3
vpc = D['xyz_vpc']


def voxel_match(xyz):
    xyz_new = np.zeros((xyz.shape[0], voxel_num, voxel_num, voxel_num, channel_num))
    for p in range(vpc.shape[0]):
        xyz_new[:, vpc[p, 0], vpc[p, 1], vpc[p, 2], :] = xyz[:, p, :]
    return xyz_new


# 训练集
train_in = np.concatenate((train_y, gan_y[:ratio_add[num]]), axis=0)
train_in = voxel_match(train_in)
train_out = np.concatenate((train_x, gan_x[:ratio_add[num]]), axis=0)
# 验证集
test_in = voxel_match(test_y)
test_out = test_x

""" part5 不同样本量下的模型训练 """
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
        path, label)
    
    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + label + '/result.npz', e_m=e_m, a_m=a_m)


# 训练集
train_in = np.concatenate((train_y, gan_y[:ratio_add[num]]), axis=0)
train_out = np.concatenate((train_x, gan_x[:ratio_add[num]]), axis=0)
# 验证集
test_in = test_y
test_out = test_x


""" 1DCNN """
label = '1DCNN_' + ratio_no[num]
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
        path, label)

    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + label + '/result.npz', e_m=e_m, a_m=a_m)


""" ANN """
label = 'ANN_' + ratio_no[num]
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
        path, label)

    # 保存结果
    e_m.append(eval_metrics.values)
    a_m.append(accuracy_metrics)
    # 清除一次内存
    gc.collect()

print(np.mean(np.mean(e_m, axis=0), axis=0))
print(np.mean(np.std(e_m, axis=0), axis=0))
print(np.mean(a_m, axis=0))
print(np.std(a_m, axis=0))

np.savez(path + label + '/result.npz', e_m=e_m, a_m=a_m)


# T = np.load('GAN2023/results/Z1/3DCNN_0.50/result.npz')
# e_m = T['e_m']
# a_m = T['a_m']
#
# e_m_1 = np.delete(e_m, [ ], axis=0)
# a_m_1 = np.delete(a_m, [ ], axis=0)
#
# print(np.mean(np.mean(e_m_1, axis=0), axis=0))
# print(np.mean(np.std(e_m_1, axis=0), axis=0))
# print(np.mean(a_m_1, axis=0))
# print(np.std(a_m_1, axis=0))
