# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(r'/root/autodl-tmp/Pointcloud2023/')
# os.chdir('/root/autodl-tmp/Pointcloud2023/')
import numpy as np
from sklearn.decomposition import PCA
from GAN2023.Z1_classes import Metrics
metrics = Metrics()


""" part1 导入真实数据、生成数据 """
# 导入真实数据
D = np.load('GAN2023/datasets/all_data.npz')
train_x, train_y = D['train_o'], D['train_xyz']
test_x, test_y = D['test_o'], D['test_xyz']
# 导入生成数据
T = np.load('GAN2023/datasets/random_0.npz')
gan_x = T['random']
TT = np.load('GAN2023/datasets/random_0_gan.npz')
gan_y = TT['gan']

""" part2 确定样本比例 """
# 正样本比例
ratio_yes = np.linspace(0.95, 0.50, 10)
# 负样本添加数量
ratio_add = np.ceil(475 / ratio_yes - 500).astype(int)
# 负样本比例
ratio_no = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50']

# ###################################################第 num 次试验
num = 7
# ###################################################第 num 次试验

# 训练集
train_in = np.concatenate((train_y, gan_y[:ratio_add[num]]), axis=0)
train_out = np.concatenate((train_x, gan_x[:ratio_add[num]]), axis=0)
# 验证集
test_in = test_y
test_out = test_x


# 点云降维
X = np.concatenate((train_in.reshape(train_in.shape[0], -1),
                    test_in.reshape(test_in.shape[0], -1)), axis=0)
pca = PCA(n_components=14)
XX = pca.fit_transform(X)
# np.sum(pca.explained_variance_ratio_)
# 选择数据
train_input = XX[:train_in.shape[0], :]
train_output = train_out
test_input = XX[train_in.shape[0]:, :]
test_output = test_out


""" 支持向量回归 SVR """
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

clf = MultiOutputRegressor(SVR(kernel='rbf')).fit(train_input, train_output)

y_pred = clf.predict(test_input)
y_real = test_output
eval_metrics_SVR = metrics.metrics_eval(y_pred, y_real)
accuracy_metrics_SVR = metrics.percentage_eval(y_pred, y_real)

print(np.mean(eval_metrics_SVR.values, axis=0))
print(accuracy_metrics_SVR)


""" 岭回归 RR """
from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0).fit(train_input, train_output)

y_pred = clf.predict(test_input)
y_real = test_output
eval_metrics_RR = metrics.metrics_eval(y_pred, y_real)
accuracy_metrics_RR = metrics.percentage_eval(y_pred, y_real)

print(np.mean(eval_metrics_RR.values, axis=0))
print(accuracy_metrics_RR)


""" Lasso回归 """
from sklearn.linear_model import MultiTaskLasso

clf = MultiTaskLasso(alpha=0.1).fit(train_input, train_output)

y_pred = clf.predict(test_input)
y_real = test_output
eval_metrics_LASSO = metrics.metrics_eval(y_pred, y_real)
accuracy_metrics_LASSO = metrics.percentage_eval(y_pred, y_real)

print(np.mean(eval_metrics_LASSO.values, axis=0))
print(accuracy_metrics_LASSO)


# """ 核岭回归 KRR """
# from sklearn.kernel_ridge import KernelRidge
#
# clf = KernelRidge(kernel='rbf').fit(train_input, train_output)
#
# y_pred = clf.predict(test_input)
# y_real = test_output
# eval_metrics_KRR = metrics.metrics_eval(y_pred, y_real)
# accuracy_metrics_KRR = metrics.percentage_eval(y_pred, y_real)
#
# print(np.mean(eval_metrics_KRR.values, axis=0))
# print(accuracy_metrics_KRR)


# """ 随机森林 RF """
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
#
# clf = RandomForestRegressor(n_estimators=50, max_depth=2, random_state=0).fit(train_input, train_output)
#
# y_pred = clf.predict(test_input)
# y_real = test_output
# eval_metrics_RF = metrics.metrics_eval(y_pred, y_real)
# accuracy_metrics_RF = metrics.percentage_eval(y_pred, y_real)
#
# print(np.mean(eval_metrics_RF.values, axis=0))
# print(accuracy_metrics_RF)


""" 梯度提升树 GBT """
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

clf = MultiOutputRegressor(GradientBoostingRegressor()).fit(train_input, train_output)

y_pred = clf.predict(test_input)
y_real = test_output
eval_metrics_GBT = metrics.metrics_eval(y_pred, y_real)
accuracy_metrics_GBT = metrics.percentage_eval(y_pred, y_real)

print(np.mean(eval_metrics_GBT.values, axis=0))
print(accuracy_metrics_GBT)


""" 高斯过程回归 GPR """
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()
clf = GaussianProcessRegressor(kernel=kernel).fit(train_input, train_output)

y_pred = clf.predict(test_input)
y_real = test_output
eval_metrics_GPR = metrics.metrics_eval(y_pred, y_real)
accuracy_metrics_GPR = metrics.percentage_eval(y_pred, y_real)

print(np.mean(eval_metrics_GPR.values, axis=0))
print(accuracy_metrics_GPR)


a = [np.mean(eval_metrics_SVR.values, axis=0),
     np.mean(eval_metrics_RR.values, axis=0),
     np.mean(eval_metrics_LASSO.values, axis=0),
     # np.mean(eval_metrics_KRR.values, axis=0),
     # np.mean(eval_metrics_RF.values, axis=0),
     np.mean(eval_metrics_GBT.values, axis=0),
     np.mean(eval_metrics_GPR.values, axis=0)]
aa = np.asarray(a)
print(aa)

b = [accuracy_metrics_SVR,
     accuracy_metrics_RR,
     accuracy_metrics_LASSO,
     # accuracy_metrics_KRR,
     # accuracy_metrics_RF,
     accuracy_metrics_GBT,
     accuracy_metrics_GPR]
bb = np.asarray(b)
print(bb)
