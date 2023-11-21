import pyldpc
import numpy as np
from tensorflow import keras
import os

import autoencoder
import myldpc
import utils

# 初始化ldpc码的H和G矩阵
baseGraph = np.mat([[0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -
                        1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -
                     1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -
                     1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                    [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -
                     1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
                    [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -
                     1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                    [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1,
                     0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                    [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -
                     1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                    [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -
                     1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                    [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -
                     1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                    [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -
                     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                    [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])
blockSize = int(27)
infoLen = int(324)
codeLen = int(648)

H = myldpc.ldpcQC(blockSize, baseGraph)
G = pyldpc.coding_matrix(H)

print('ldpc init done.')

# 初始化Bob和Eve的snr
BobSnr = 100
EveSnr = 1
Bob_noise_sigma = utils.snr_to_noise(BobSnr)
Eve_noise_sigma = utils.snr_to_noise(EveSnr)
# 初始化模型参数
innerLen = 24
modLen = 4
outerLen = innerLen // modLen
learning_rate = 0.01

# 检测是否已训练模型
Bob_model_path_1 = "Bob_autoencoder_1.keras"
Eve_model_path_1 = "Eve_autoencoder_1.keras"
Bob_trained_1 = False
Eve_trained_1 = False
if os.path.exists(Bob_model_path_1):
    Bob_autoencoder = keras.models.load_model(Bob_model_path_1)
    Bob_trained_1 = True
else:
    Bob_autoencoder = autoencoder.init_autoencoder(
        Bob_noise_sigma, innerLen, outerLen)

if os.path.exists(Eve_model_path_1):
    Eve_autoencoder = keras.models.load_model(Eve_model_path_1)
    Eve_trained_1 = True
else:
    Eve_autoencoder = autoencoder.init_autoencoder(
        Eve_noise_sigma, innerLen, outerLen)

print('autoencoder init done.')
# 定义学习率调度器


def lr_scheduler(epoch, lr):
    if (epoch + 1) % 50 == 0:
        return lr * 0.5  # 每50个epoch学习率减半
    else:
        return lr


# 定义学习率调度器回调
lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

# 第一轮训练
# 生成训练集
SampleSize = 1024
SampleData = np.random.randint(0, 2, (SampleSize, innerLen))

# 训练
nEpochs = 100
BatchSize = 120
if Bob_trained_1 == False:
    print('Bob_autoencoder_1 is training...')
    Bob_avg_losses = autoencoder.train_autoencoder(
        nEpochs, BatchSize, learning_rate, lr_scheduler, SampleData, Bob_autoencoder)
    print('Bob_autoencoder_1 is trained.')
    # 显示损失函数
    Bob_loss_fig = autoencoder.show_losses(nEpochs, Bob_avg_losses)
    # 保存损失函数图像
    Bob_loss_fig.savefig('Bob_loss_1.png')
    # 保存模型
    Bob_autoencoder.save(
        filepath='Bob_autoencoder_1.keras', save_format='keras')

if Eve_trained_1 == False:
    print('Eve_autoencoder_1 is training...')
    Eve_avg_losses = autoencoder.train_autoencoder(
        nEpochs, BatchSize, learning_rate, lr_scheduler, SampleData, Eve_autoencoder)
    print('Eve_autoencoder_1 is trained.')
    # 显示损失函数
    Eve_loss_fig = autoencoder.show_losses(nEpochs, Eve_avg_losses)
    # 保存损失函数图像
    Eve_loss_fig.savefig('Eve_loss_1.png')
    # 保存模型
    Eve_autoencoder.save(
        filepath='Eve_autoencoder_1.keras', save_format='keras')

# 第一轮SER测试
# 生成测试集
TestSize = SampleSize * 10
TestData = np.random.randint(0, 2, (TestSize, innerLen))
Bob_Predict = Bob_autoencoder.predict(TestData)
Eve_Predict = Eve_autoencoder.predict(TestData)
Bob_MSE = np.sum(
    np.square(np.abs(np.subtract(Bob_Predict, TestData))))/TestSize/innerLen
Eve_MSE = np.sum(
    np.square(np.abs(np.subtract(Eve_Predict, TestData))))/TestSize/innerLen

print('Bob_MSE_1=', Bob_MSE)
print('Eve_MSE_1=', Eve_MSE)

Bob_Data = np.where(Bob_Predict >= 0.5, 1, 0)
Eve_Data = np.where(Eve_Predict >= 0.5, 1, 0)
Bob_Ser = np.sum(np.abs(np.subtract(Bob_Data, TestData)))/TestSize/innerLen
Eve_Ser = np.sum(np.abs(np.subtract(Eve_Data, TestData)))/TestSize/innerLen

print('Bob_Ser_1=', Bob_Ser)
print('Eve_Ser_1=', Eve_Ser)

# 物理层安全训练
