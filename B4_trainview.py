# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {'font.family': 'Times New Roman',
          'font.size': 25}
rcParams.update(config)


# 绘图
class Trainview:
    def __init__(self):
        self.s = 0.2
        self.v = 1

    def plot_a(self, ax, xyz, xyz_, num):
        a = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=self.s, marker='o',
                       c=xyz_[:, num], vmin=-self.v, vmax=self.v, cmap='bwr')
        return a

    @staticmethod
    def plot_ax(ax):
        ax.view_init(elev=15., azim=60.)
        ax.axis('off')
        ax.set_xlim(2000, 3000)
        ax.set_ylim(0, 1000)
        ax.set_zlim(1500, 2500)
        return ax

    # 真实状态、预测状态对比，3*2，左真实，右预测
    def training_plot(self, xyz, model, test_x, test_target, xyz_h, xyz_c, 
                      path=None, epoch=None):
        
        test_pred = model(test_x, training=True).numpy()
        
        xyz_pred_c_1 = test_pred[0, :]
        xyz_pred_c = xyz_pred_c_1 * xyz_c + xyz_h

        xyz_real_c_1 = test_target[0, :]
        xyz_real_c = xyz_real_c_1 * xyz_c + xyz_h

        # 绘制
        fig = plt.figure(figsize=(5, 6), dpi=300)
        
        ax1 = fig.add_subplot(321, projection='3d')
        a1 = self.plot_a(ax1, xyz, xyz_real_c, 0)
        ax1 = self.plot_ax(ax1)
        
        ax2 = fig.add_subplot(322, projection='3d')
        a2 = self.plot_a(ax2, xyz, xyz_pred_c, 0)
        ax2 = self.plot_ax(ax2)
        
        ax3 = fig.add_subplot(323, projection='3d')
        a3 = self.plot_a(ax3, xyz, xyz_real_c, 1)
        ax3 = self.plot_ax(ax3)
        
        ax4 = fig.add_subplot(324, projection='3d')
        a4 = self.plot_a(ax4, xyz, xyz_pred_c, 1)
        ax4 = self.plot_ax(ax4)
        
        ax5 = fig.add_subplot(325, projection='3d')
        a5 = self.plot_a(ax5, xyz, xyz_real_c, 2)
        ax5 = self.plot_ax(ax5)
        
        ax6 = fig.add_subplot(326, projection='3d')
        a6 = self.plot_a(ax6, xyz, xyz_pred_c, 2)
        ax6 = self.plot_ax(ax6)

        fig.colorbar(a1, ax=[ax1, ax2, ax3, ax4, ax5, ax6])
        
        l1_loss = tf.reduce_mean(tf.abs(xyz_real_c_1 - xyz_pred_c_1))

        if path is not None and epoch is not None:
            plt.savefig(path + str(epoch.numpy()+1) + '_' + str(round(l1_loss.numpy(), 6)) + '.png')

        plt.show()
        print(l1_loss)

    # X、Y、Z向偏差状态
    def xyz_c_plot(self, xyz, xyz_c, index=None):
        fig = plt.figure(figsize=(6, 2), dpi=300)
        
        ax1 = fig.add_subplot(131, projection='3d')
        a1 = self.plot_a(ax1, xyz, xyz_c, 0)
        ax1 = self.plot_ax(ax1)
        
        ax2 = fig.add_subplot(132, projection = '3d')
        a2 = self.plot_a(ax2, xyz, xyz_c, 1)
        ax2 = self.plot_ax(ax2)
        
        ax3 = fig.add_subplot(133, projection = '3d')
        a3 = self.plot_a(ax3, xyz, xyz_c, 2)
        ax3 = self.plot_ax(ax3)
        
        # fig.colorbar(a1, ax=[ax1,ax2,ax3])
        
        if index is not None:
            plt.savefig('C:/Users/ruiruiruidian/Desktop/123/' + str(index) + '.png')
        plt.show()



if __name__ == '__main__':
    import numpy as np
    draw = Trainview()

    D = np.load('GAN2023/datasets/all_data.npz')
    train_in, train_out = D['train_xyz'], D['train_o']
    test_in, test_out = D['test_xyz'], D['test_o']
    xyz = D['xyz_nominal']
    for index in range(train_in.shape[0]):
        draw.xyz_c_plot(xyz, train_in[index, :], index)



