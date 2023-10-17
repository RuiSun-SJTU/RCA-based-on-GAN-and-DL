# -*- coding: utf-8 -*-
# 绘制WGANGP训练过程中MAE和GP的变化情况
# 绘制DCGAN、LSGAN、WGANGP训练过程中MAE的变化情况
# 计算不同所有模型生成点云的 psnr、ssim、nrmse、mae
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {'font.family': 'Times New Roman',
          'font.size': 25,
          'mathtext.fontset': 'stix',
          'font.serif': ['SimSun'],
          'xtick.direction': 'in',
          'ytick.direction': 'in'
          }
rcParams.update(config)


Path = 'GAN2023/model/'
losses = ['wgangp', 'dcgan', 'lsgan']
nets = ['yes', 'no']
l1s = ['L1', 'L2', 'N']
metrics = ['losss', 'Mae', 'psnr', 'ssim', 'nrmse', 'mae_']

D = dict()
for loss in losses:
    for net in nets:
        for l1 in l1s:
            for metric in metrics:
                name = loss + '_' + net + '_' + l1 + '_' + metric
                path = Path + loss + '_' + net + '_' + l1 + '/' + metric + '.csv'
                D[name] = np.loadtxt(path, delimiter=',')


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 18), dpi=100)

ind = np.linspace(25, 5000, 200, dtype=int)
index = np.linspace(0, 5000, 11, dtype=int)

ax[0].plot(ind, D['wgangp_no_N_losss'][:, 6], ls='--', color='royalblue', lw=2.5, label='wgangp_no_N')
ax[0].plot(ind, D['wgangp_no_L1_losss'][:, 6], ls='--', color='orangered', lw=2.5, label='wgangp_no_L1')
ax[0].plot(ind, D['wgangp_yes_N_losss'][:, 6], ls='-', color='royalblue', lw=2.5, label='wgangp_yes_N')
ax[0].plot(ind, D['wgangp_yes_L1_losss'][:, 6], ls='-', color='orangered', lw=2.5, label='wgangp_yes_L1')

ax[0].set_xlabel('Epochs', fontsize=30)
ax[0].set_ylabel('Gradient penalty', fontsize=30)
ax[0].set_xticks(index)
ax[0].set_xlim([0, 5000])
ax[0].set_ylim([0.0, 0.4])
ax[0].tick_params(width=2.0, labelsize=25)
ax[0].legend(loc=1, ncol=1, prop={'size': 25})

ax[0].spines['top'].set_linewidth(2)
ax[0].spines['right'].set_linewidth(2)
ax[0].spines['bottom'].set_linewidth(2)
ax[0].spines['left'].set_linewidth(2)

ax[1].plot(ind, D['wgangp_no_N_Mae'], ls='--', color='royalblue', lw=2.5, label='wgangp_no_N')
ax[1].plot(ind, D['wgangp_no_L1_Mae'], ls='--', color='orangered', lw=2.5, label='wgangp_no_L1')
ax[1].plot(ind, D['wgangp_yes_N_Mae'], ls='-', color='royalblue', lw=2.5, label='wgangp_yes_N')
ax[1].plot(ind, D['wgangp_yes_L1_Mae'], ls='-', color='orangered', lw=2.5, label='wgangp_yes_L1')

ax[1].set_xlabel('Epochs', fontsize=30)
ax[1].set_ylabel('Mean Absolute Error', fontsize=30)
ax[1].set_xticks(index)
ax[1].set_xlim([0, 5000])
ax[1].set_ylim([0.0, 0.4])
ax[1].tick_params(width=2.0, labelsize=25)
ax[1].legend(loc=1, ncol=1, prop={'size': 25})

ax[1].spines['top'].set_linewidth(2)
ax[1].spines['right'].set_linewidth(2)
ax[1].spines['bottom'].set_linewidth(2)
ax[1].spines['left'].set_linewidth(2)

fig.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 18), dpi=100)


def draw(r, rs, xs, ss, cs):
    for x, s, c in zip(xs, ss, cs):
        ax[r].plot(ind, D[x+'_Mae'], ls=s, color=c, lw=2.5, label=x)

    ax[r].set_title(rs, x=-0.10, y=1.05, fontsize=30)
    ax[r].set_xlabel('Epochs', fontsize=30)
    ax[r].set_ylabel('Mean Absolute Error', fontsize=30)
    ax[r].set_xticks(index)
    ax[r].set_xlim([0, 5000])
    ax[r].set_ylim([0.0, 1.25])
    ax[r].tick_params(width=2.0, labelsize=25)
    ax[r].legend(loc=9, ncol=4, prop={'size': 22})

    ax[r].spines['top'].set_linewidth(2)
    ax[r].spines['right'].set_linewidth(2)
    ax[r].spines['bottom'].set_linewidth(2)
    ax[r].spines['left'].set_linewidth(2)


ind = np.linspace(25, 5000, 200, dtype=int)
index = np.linspace(0, 5000, 11, dtype=int)
wgangp = ['wgangp_yes_L1', 'wgangp_yes_N', 'wgangp_no_L1', 'wgangp_no_N']
# wgangp = ['wgangp_yes_L1', 'lsgan_yes_L1', 'dcgan_yes_L1']
lsgan = ['lsgan_yes_L1', 'lsgan_yes_N', 'lsgan_no_L1', 'lsgan_no_N']
dcgan = ['dcgan_yes_L1', 'dcgan_yes_N', 'dcgan_no_L1', 'dcgan_no_N']
SS = ['-', '-', '--', '--']
CS = ['blue', 'red', 'blue', 'red']
draw(0, '(a)', wgangp, SS, CS)
draw(1, '(b)', lsgan, SS, CS)
draw(2, '(c)', dcgan, SS, CS)

fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()


data = np.zeros((4, 18))
index_1, index_2 = 0, 0
metrics_ = ['psnr', 'ssim', 'nrmse', 'mae_']

for loss in losses:
    for net in nets:
        for l1 in l1s:
            for metric_ in metrics_:
                data[index_1, index_2] = D[loss + '_' + net + '_' + l1 + '_' + metric_][50:].mean()
                index_1 += 1
            index_2 += 1
            index_1 = 0

value = []
for loss in losses:
    for net in nets:
        for l1 in l1s:
            value.append(D[loss + '_' + net + '_' + l1 + '_' + 'mae_'])

val = np.transpose(np.asarray(value))

a = np.where(np.argmin(val, axis=1) == 0)[0]


