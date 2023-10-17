# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(r'/root/autodl-tmp/Pointcloud2023/')
# os.chdir('/root/autodl-tmp/Pointcloud2023/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {'font.family': 'Times New Roman',
          'mathtext.fontset': 'stix',
          'font.serif': ['SimSun'],
          'xtick.direction': 'in',
          'ytick.direction': 'in'
          }
rcParams.update(config)


def data_in(path, x):
    e, a = {}, {}
    for ratio in x:
        t = np.load(path + ratio + '/result.npz')
        e[ratio] = t['e_m']
        a[ratio] = t['a_m']
    return e, a


def mean_std(t_e, t_a):
    m, s = {}, {}

    m['e'] = list(map(lambda x: x.mean(axis=0).mean(axis=0), t_e.values()))
    m['MAE'] = list(map(lambda x: x[0], m['e']))
    m['MSE'] = list(map(lambda x: x[1], m['e']))
    m['RMSE'] = list(map(lambda x: x[2], m['e']))
    m['R2'] = list(map(lambda x: x[3], m['e']))
    
    m['a'] = list(map(lambda x: x.mean(axis=0), t_a.values()))
    m['Accuracy'] = list(map(lambda x: x[0], m['a']))
    m['Precision'] = list(map(lambda x: x[1], m['a']))
    m['Recall'] = list(map(lambda x: x[2], m['a']))
    m['F1_score'] = list(map(lambda x: x[3], m['a']))
    
    s['e'] = list(map(lambda x: x.std(axis=0).mean(axis=0), t_e.values()))
    s['MAE'] = list(map(lambda x: x[0], s['e']))
    s['MSE'] = list(map(lambda x: x[1], s['e']))
    s['RMSE'] = list(map(lambda x: x[2], s['e']))
    s['R2'] = list(map(lambda x: x[3], s['e']))
    
    s['a'] = list(map(lambda x: x.std(axis=0), t_a.values()))
    s['Accuracy'] = list(map(lambda x: x[0], s['a']))
    s['Precision'] = list(map(lambda x: x[1], s['a']))
    s['Recall'] = list(map(lambda x: x[2], s['a']))
    s['F1_score'] = list(map(lambda x: x[3], s['a']))
    
    return m, s


Path = 'GAN2023/results/Z1/3DCNN_'
X = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45']
T_e, T_a = data_in(Path, X)

M, S = mean_std(T_e, T_a)

c_b = 'cornflowerblue'
c_r = 'salmon'
color_0 = [c_b, c_b, c_b, c_b, c_b, c_b, c_r, c_b, c_b]
color_1 = [c_b, c_b, c_b, c_b, c_b, c_b, c_r, c_b, c_b]
color_2 = [c_b, c_b, c_b, c_b, c_b, c_b, c_b, c_r, c_b]
color_3 = [c_b, c_b, c_b, c_b, c_r, c_b, c_b, c_b, c_b]
color_4 = [c_b, c_b, c_b, c_b, c_b, c_b, c_b, c_r, c_b]
color_5 = [c_b, c_b, c_b, c_b, c_b, c_b, c_b, c_r, c_b]

error_kw_ = {'ecolor': '0.5', 'capsize': 5}
ind = np.arange(len(X))
width = 0.6


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), dpi=100)


def draw(r, c, s, color_, num, min_, max_):
    p = ax[r, c].bar(ind, M[s], width=width, color=color_,
                     yerr=np.asarray(S[s])/2, error_kw=error_kw_, label=s)

    # 设置标题、X、Y轴刻度
    ax[r, c].set_xlabel('Proportion of failure samples', fontsize=23)
    ax[r, c].set_ylabel(s, fontsize=23)
    ax[r, c].set_title(num, x=-0.13, y=1.05, fontsize=25)
    ax[r, c].set_xticks(ind, labels=X)
    ax[r, c].set_ylim([min_, max_])
    ax[r, c].tick_params(width=1.5, labelsize=20)
    
    # 去掉边框
    ax[r, c].spines['top'].set_visible(False)
    ax[r, c].spines['right'].set_visible(False)
    ax[r, c].spines['bottom'].set_linewidth(1.5)
    ax[r, c].spines['left'].set_linewidth(1.5)
    
    # 设置标签
    ax[r, c].bar_label(p, fmt='%.4f', fontsize=15, color='k')


draw(r=0, c=0, s='MAE',         color_=color_0, num='(a)', min_=0.04, max_=0.06)
draw(r=0, c=1, s='R2',          color_=color_1, num='(b)', min_=0.98, max_=0.99)
draw(r=1, c=0, s='Accuracy',    color_=color_2, num='(c)', min_=0.90, max_=1.0)
draw(r=1, c=1, s='Precision',   color_=color_3, num='(d)', min_=0.94, max_=1.0)
draw(r=2, c=0, s='Recall',      color_=color_4, num='(e)', min_=0.85, max_=1.0)
draw(r=2, c=1, s='F1_score',    color_=color_5, num='(f)', min_=0.90, max_=1.0)

fig.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()
