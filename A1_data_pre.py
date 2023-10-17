# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# ## 全部数据导入
class GetData:

    # 读取输入输出文件 用pandas模块读取csv格式文件
    @staticmethod
    def data_import(file_names, file_path):
        data = []
        for file_name in file_names:
            file = file_path + file_name
            data.append(pd.read_csv(file, header=None))
        dataset = pd.concat(data, ignore_index=True)  # 数据合并
        return dataset

    # 读取体素化点云 dat格式文件
    @staticmethod
    def data_voxel(file_names, file_path):
        file = file_path + file_names
        try:
            vpc = np.load(file, allow_pickle=True)
            return vpc
        except AssertionError as error:
            print(error)
            print('Voxel_Mapping File not found !')

    # 读取点云坐标值 用numpy模块读取csv格式文件
    @staticmethod
    def data_nominal(file_names, file_path):
        file = file_path + file_names
        try:
            xyz_ = np.loadtxt(file, delimiter=',')
            return xyz_
        except AssertionError as error:
            print(error)
            print('Nominal_Cop File not found !')

    # 尺寸偏差串联 n*10841*3
    @staticmethod
    def x_y_z_var_merge(x_old, y_old, z_old):
        xyz_var = np.concatenate((x_old.values[:, :, np.newaxis],
                                  y_old.values[:, :, np.newaxis],
                                  z_old.values[:, :, np.newaxis]),
                                 axis=2)
        return xyz_var


# ## 读取数据
class DataSample:

    def __init__(self):
        import tensorflow as tf
        self.D = np.load('GAN2023/datasets/all_data.npz')

        self.train_x = self.D['train_o'].astype('float32')
        self.train_y = self.D['train_xyz'].astype('float32')
        self.test_x = self.D['test_o'].astype('float32')
        self.test_y = self.D['test_xyz'].astype('float32')
        self.xyz_h = self.D['xyz_h']
        self.xyz_c = self.D['xyz_c']
        self.xyz_vpc = self.D['xyz_vpc']
        self.xyz_nominal = self.D['xyz_nominal']

        self.train_xy = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))

    def data_in(self, batch):
        train_xy = self.train_xy.shuffle(self.train_x.shape[0], reshuffle_each_iteration=True).batch(batch)

        return self.xyz_vpc, self.xyz_nominal, train_xy, \
            self.test_x, self.test_y, self.xyz_h, self.xyz_c

    def data_in_(self, batch):
        train_xy = self.train_xy.shuffle(self.train_x.shape[0], reshuffle_each_iteration=True).batch(batch)

        return self.xyz_vpc, self.xyz_nominal, train_xy, \
            self.train_x, self.train_y, self.test_x, self.test_y, self.xyz_h, self.xyz_c


if __name__ == '__main__':
    """ part1: 原始数据导入与保存 """
    get_data = GetData()

    # 文件地址
    file_path_load = 'datasets/inner_rf_assembly/'
    file_path_save = 'datasets/'
    # 尺寸偏差
    import_file_name_x = ['output_table_x.csv', 'test_output_table_x.csv']
    import_file_name_y = ['output_table_y.csv', 'test_output_table_y.csv']
    import_file_name_z = ['output_table_z.csv', 'test_output_table_z.csv']
    dataset_import_x = get_data.data_import(import_file_name_x, file_path_load)
    dataset_import_y = get_data.data_import(import_file_name_y, file_path_load)
    dataset_import_z = get_data.data_import(import_file_name_z, file_path_load)
    data_import = get_data.x_y_z_var_merge(dataset_import_x, dataset_import_y, dataset_import_z)
    # 工艺参数
    output_file_names = ['input_X.csv', 'test_input_X.csv']
    data_output = get_data.data_import(output_file_names, file_path_load)
    # 体素化网格
    vpc_file_names = 'inner_rf_64_voxel_mapping.dat'
    dataset_vpc = get_data.data_voxel(vpc_file_names, file_path_load)
    xyz_vpc = dataset_vpc.astype(int)
    # 点云名义位置
    xyz_file_names = 'inner_rf_nominal_cop.csv'
    xyz_nominal = get_data.data_nominal(xyz_file_names, file_path_load)

    # 保存输入输出、vpc、nominal
    np.savez(file_path_save + 'data_xyz_o', data_import=data_import, data_output=data_output)
    np.savez(file_path_save + 'data_vpc_nominal', xyz_vpc=xyz_vpc, xyz_nominal=xyz_nominal)

    """ part2: 挑选数据集 """
    import numpy as np

    D = np.load(r'D:\Anaconda\ganpoint\datasets\data_xyz_o.npz')
    data_import = D['data_import']
    data_output = D['data_output']
    T = np.load(r'D:\Anaconda\ganpoint\datasets\data_vpc_nominal.npz')
    xyz_vpc = T['xyz_vpc']
    xyz_nominal = T['xyz_nominal']

    # ###
    a = data_output
    min_in = 0.98
    max_in = 0.98

    c0 = np.where((abs(a[:, 0]) < min_in) & (abs(a[:, 1]) < min_in) & (abs(a[:, 2]) < min_in) &
                  (abs(a[:, 3]) < max_in) & (abs(a[:, 4]) < max_in) & (abs(a[:, 5]) < max_in) &
                  (data_import[:, -1, 0] == 1))[0]

    c7 = np.where((abs(a[:, 0]) < min_in) & (abs(a[:, 1]) < min_in) & (abs(a[:, 2]) < min_in) &
                  (a[:, 3] < -1.01) & (a[:, 3] > -1.50) &
                  (abs(a[:, 4]) < max_in) & (abs(a[:, 5]) < max_in) &
                  (data_import[:, -1, 0] == 1))[0]

    from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering
    from collections import Counter

    e1 = a[c0]
    e2 = SpectralClustering(n_clusters=3).fit(e1)
    Counter(e2.labels_).most_common()
    e5 = e1[np.where(e2.labels_ == 0)]
    e6 = e1[np.where(e2.labels_ == 1)]
    e7 = e1[np.where(e2.labels_ == 2)]

    e3 = c0[np.where((e2.labels_ == 1) | (e2.labels_ == 2))]

    f1 = a[c7]
    f2 = SpectralClustering(n_clusters=2).fit(f1)
    Counter(f2.labels_).most_common()
    f5 = f1[np.where(f2.labels_ == 0)]
    f6 = f1[np.where(f2.labels_ == 1)]

    ff5 = np.delete(f5, 3, axis=1)
    ff6 = np.delete(f6, 3, axis=1)

    f3 = c7[np.where(f2.labels_ == 0)]

    # ###
    b0 = list(e3.copy())
    np.random.shuffle(b0)
    d0 = np.random.choice(b0, 525, replace=False)
    d0 = list(d0.copy())
    b7 = list(f3.copy())
    np.random.shuffle(b7)
    d7 = np.random.choice(b7, 75, replace=False)
    d7 = list(d7.copy())

    num_train = d0[:475] + d7[:25]
    num_test = d0[475:] + d7[25:]

    train_xyz_ = data_import[num_train, :-1, :]
    train_o = data_output[num_train, :]
    test_xyz_ = data_import[num_test, :-1, :]
    test_o = data_output[num_test, :]

    # ###
    xyz = np.concatenate((train_xyz_, test_xyz_), axis=0)
    xyz_h = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    xyz_c = (xyz.max(axis=0) - xyz.min(axis=0)) / 2

    train_xyz = (train_xyz_ - xyz_h) / xyz_c
    test_xyz = (test_xyz_ - xyz_h) / xyz_c

    # ###
    np.savez('GAN2022/datasets/all_data.npz',
             train_o=train_o, train_xyz=train_xyz,
             test_o=test_o, test_xyz=test_xyz,
             xyz_h=xyz_h, xyz_c=xyz_c,
             xyz_vpc=xyz_vpc, xyz_nominal=xyz_nominal)

    """
    ### 聚类
    from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering
    from collections import Counter

    e1 = a[c0,:]
    #k = AffinityPropagation(random_state=10).fit(d)
    e2 = KMeans(n_clusters=2).fit(e1)
    #e2 = SpectralClustering(n_clusters=2).fit(e1)
    Counter(e2.labels_).most_common()
    e5 = e1[np.where(e2.labels_== 0)]


    D = np.load('GAN2022/datasets/all_data.npz')
    train_val_o = D['train_val_o']          # 1000*6
    train_val_xyz = D['train_val_xyz']      # 1000*10841*3
    train_o = D['train_o']                  # 800*6
    train_xyz = D['train_xyz']              # 800*10841*3
    val_o = D['val_o']                      # 200*6
    val_xyz = D['val_xyz']                  # 200*10841*3
    test_o = D['test_o']                    # 100*6
    test_xyz = D['test_xyz']                # 100*10841*3
    xyz_h = D['xyz_h']                      # 10841*3
    xyz_c = D['xyz_c']                      # 10841*3
    xyz_vpc = D['xyz_vpc']                  # 10841*3
    xyz_nominal = D['xyz_nominal']          # 10841*3
    """
