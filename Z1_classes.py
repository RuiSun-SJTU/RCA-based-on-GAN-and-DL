# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class Model:

    def __init__(self, reg_coeff, dro_coeff, output_num, output_type, opt, loss):
        self.reg_coeff = reg_coeff
        self.dro_coeff = dro_coeff
        self.output_num = output_num
        self.output_type = output_type
        self.opt = opt
        self.loss = loss

    # 原论文代码
    def cnn_3d_(self, voxel_num, channel_num):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Conv3D, MaxPool3D
        from tensorflow.keras.layers import Flatten, Dense

        model = Sequential()
        model.add(Conv3D(32, 5, 2, activation='relu', input_shape=(voxel_num, voxel_num, voxel_num, channel_num)))
        model.add(Conv3D(32, 4, 2, activation='relu'))
        model.add(Conv3D(32, 3, 1, activation='relu'))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))
        model.add(Flatten())
        model.add(Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu'))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu'))
        model.add(Dense(self.output_num, activation=self.output_type))
        model.compile(optimizer=self.opt, loss=self.loss, metrics=['mae'])

        return model

    # 加了BN之后的代码
    def cnn_3d(self, voxel_num, channel_num):
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, LeakyReLU, MaxPool3D
        from tensorflow.keras.layers import Flatten, Dense

        x = Input(shape=(voxel_num, voxel_num, voxel_num, channel_num))
        y = Conv3D(32, 5, 2)(x)
        y = LeakyReLU()(y)
        y = Conv3D(32, 4, 2)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv3D(32, 3, 1)(y)
        y = LeakyReLU()(y)
        y = MaxPool3D(pool_size=(2, 2, 2))(y)
        y = Flatten()(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        output = Dense(self.output_num, activation=self.output_type)(y)

        model = Model(inputs=x, outputs=output)
        model.compile(optimizer=self.opt, loss=self.loss, metrics=['mae'])

        return model

    # 1DCNN
    def cnn_1d(self, voxel_num, channel_num):
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D
        from tensorflow.keras.layers import Flatten, Dense

        x = Input(shape=(voxel_num, channel_num))
        y = Conv1D(128, 6, 3)(x)  # 3612
        # y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv1D(128, 7, 3)(y)  # 1202
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv1D(64, 6, 3)(y)  # 399
        # y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPool1D()(y)
        y = Flatten()(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        output = Dense(self.output_num, activation=self.output_type)(y)

        model = Model(inputs=x, outputs=output)
        model.compile(optimizer=self.opt, loss=self.loss, metrics=['mae'])

        return model

    # ANN
    def ann(self, voxel_num, channel_num):
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Input, BatchNormalization
        from tensorflow.keras.layers import Flatten, Dense

        x = Input(shape=(voxel_num, channel_num))
        y = Flatten()(x)
        y = Dense(16, activation='relu')(y)
        y = Dense(512, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Dense(512, activation='relu')(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        y = Dense(64, kernel_regularizer=regularizers.l2(self.reg_coeff), activation='relu')(y)
        output = Dense(self.output_num, activation=self.output_type)(y)

        model = Model(inputs=x, outputs=output)
        model.compile(optimizer=self.opt, loss=self.loss, metrics=['mae'])

        return model


# 绘图
class Trainview:

    def __init__(self):
        self.mae = 0.6
        self.loss = 2.0
        self.blank = 5
        self.length = 8
        self.wide = 10

    def training_plot(self, history, path=None, labels=None):
        import matplotlib.pyplot as plt

        mae = history.history['mae']
        val_mae = history.history['val_mae']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = len(history.history['loss'])

        plt.figure(figsize=(self.length, self.wide))

        plt.subplot(2, 1, 1)
        plt.plot(mae, label='Training')
        plt.plot(val_mae, label='Validation')
        plt.title('Model Mae')
        plt.xlabel('epoch')
        plt.xlim([-self.blank, epochs + self.blank])
        plt.ylabel('Mae')
        plt.ylim([0, self.mae])
        plt.legend(loc=1)

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training')
        plt.plot(val_loss, label='Validation')
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.xlim([-self.blank, epochs + self.blank])
        plt.ylabel('Loss')
        plt.ylim([0, self.loss])
        plt.legend(loc=1)

        if path is not None and labels is not None:
            mae_loss = pd.DataFrame.from_dict(history.history)
            mae_loss.to_csv(path + labels + '/mae_and_loss.csv')
            plt.savefig(path + labels + '/accuracy_loss_best.png')

        plt.show()


# 评价指标
class Metrics:

    def __init__(self):
        self.kcc = [0, 1, 2, 3, 4, 5]
        self.min = 1.0
        self.max = 1.0

    # 求预测结果于真实结果之间的 MAE\MSE\R2
    def metrics_eval(self, pred_y, real_y, path=None, label=None):
        from sklearn import metrics

        mae_kccs = metrics.mean_absolute_error(pred_y, real_y, multioutput='raw_values')
        mse_kccs = metrics.mean_squared_error(pred_y, real_y, multioutput='raw_values')
        r2_kccs = metrics.r2_score(pred_y, real_y, multioutput='raw_values')
        rmse_kccs = np.sqrt(mse_kccs)

        kcc_id = self.kcc
        eval_metrics = {
            "KCC_ID": kcc_id,
            "Mean Absolute Error": mae_kccs,
            "Mean Squared Error": mse_kccs,
            "Root Mean Squared Error": rmse_kccs,
            "R Squared": r2_kccs
        }

        accuracy_metrics = pd.DataFrame.from_dict(eval_metrics)
        accuracy_metrics = accuracy_metrics.set_index('KCC_ID')

        if path is not None and label is not None:
            accuracy_metrics.to_csv(path + label + '/metrics_train_best.csv')

        return accuracy_metrics

    # 求预测准确率的百分比
    def percentage_eval(self, pred_y, real_y):
        y1 = np.where(abs(pred_y[:, 0:3]) <= self.min, 0, 1)
        y2 = np.where(abs(pred_y[:, 3:6]) <= self.max, 0, 1)
        y3 = np.concatenate((y1, y2), axis=1)

        y4 = np.where(abs(real_y[:, 0:3]) <= self.min, 0, 1)
        y5 = np.where(abs(real_y[:, 3:6]) <= self.max, 0, 1)
        y6 = np.concatenate((y4, y5), axis=1)

        num = np.array([[0, 0], [0, 0]])

        for index in range(y6.shape[0]):
            if np.sum(y6[index, :]) == 0:
                if np.sum(abs(y6[index, :] - y3[index, :])) == 0:
                    num[0, 0] = num[0, 0] + 1
                else:
                    num[0, 1] = num[0, 1] + 1
            else:
                if np.sum(abs(y6[index, :] - y3[index, :])) == 0:
                    num[1, 1] = num[1, 1] + 1
                else:
                    num[1, 0] = num[1, 0] + 1

        accuracy = (num[0, 0] + num[1, 1]) / np.sum(num)
        precision = num[1, 1] / (num[0, 1] + num[1, 1])  # 这里的分母也包括其他错误情况
        recall = num[1, 1] / (num[1, 0] + num[1, 1])
        f1 = 2 * num[1, 1] / (np.sum(num) + num[1, 1] - num[0, 0])

        return [accuracy, precision, recall, f1]


# 总程序
class TrainModel(Trainview, Metrics):
    def __init__(self, epochs, batch_size):
        # super().__init__() 单个父类时
        Trainview.__init__(self)
        Metrics.__init__(self)
        self.epochs = epochs
        self.batch_size = batch_size

    def train_model(self, model, train_x, train_y, test_x, test_y,
                    path=None, label=None):
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.models import load_model

        # 保存最好的模型
        model_file_path = path + label + '/trained_model_best.hdf5'
        checkpointer = ModelCheckpoint(model_file_path, save_best_only=True)
        callbacks = [checkpointer]

        # 模型训练
        history = model.fit(x=train_x, y=train_y,
                            validation_data=(test_x, test_y),
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            callbacks=[callbacks])

        # 绘制损失函数
        # super().training_plot(history, path, label)

        # 验证集预测
        inference_model = load_model(model_file_path)

        y_pred = inference_model.predict(test_x)
        '''
        num = int(test_x.shape[0]/10)
        y_pred = inference_model.predict(test_x[:num, :])
        for index in range(9):
            ypred = inference_model.predict(test_x[(index+1)*num:(index+2)*num, :])
            y_pred = np.concatenate((y_pred, ypred), axis=0)
        '''
        accuracy_metrics = super().metrics_eval(y_pred, test_y, path, label)
        accuracy_percentage = super().percentage_eval(y_pred, test_y)

        print(accuracy_metrics)
        print(accuracy_percentage)

        return model, accuracy_metrics, accuracy_percentage
