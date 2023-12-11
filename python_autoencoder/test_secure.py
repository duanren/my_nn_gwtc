import pyldpc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

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

# 加载已训练模型
Alice_encoder_model_path_2 = "python_autoencoder/Alice_encoder_2.keras"
Bob_decoder_model_path_2 = "python_autoencoder/Bob_decoder_2.keras"
Eve_decoder_model_path_2 = "python_autoencoder/Eve_decoder_2.keras"

Alice_encoder = keras.models.load_model(Alice_encoder_model_path_2)
Bob_decoder = keras.models.load_model(Bob_decoder_model_path_2)
Eve_decoder = keras.models.load_model(Eve_decoder_model_path_2)

# 初始化模型参数
innerLen = 24
modLen = 4
outerLen = innerLen * 2 // modLen
print('autoencoder init done.')
# 生成测试集
TestSize = 100
Bob_snr_range = [-50,-30,-10,10, 30, 50]
Eve_snr_range = [-50,-30,-10,10, 30, 50]
Bob_BER = np.zeros(len(Bob_snr_range))
Eve_BER = np.zeros(len(Bob_snr_range))
Bob_SER = np.zeros(len(Bob_snr_range))
Eve_SER = np.zeros(len(Bob_snr_range))
Bob_MSE = np.zeros(len(Bob_snr_range))
Eve_MSE = np.zeros(len(Bob_snr_range))
for i in range(len(Bob_snr_range)):
    Bob_test_snr = Bob_snr_range[i]
    Eve_test_snr = Eve_snr_range[i]
    Bob_test_sigma = utils.snr_to_noise(Bob_test_snr)
    Eve_test_sigma = utils.snr_to_noise(Eve_test_snr)
    for j in range(TestSize):
        Info = np.random.randint(0, 2, infoLen)
        TestData = myldpc.encode(G, Info).reshape(-1, innerLen)
        Alice_code = Alice_encoder.predict(TestData)
        Bob_code = Alice_code + \
            tf.random.normal(tf.shape(Alice_code), mean=0.0,
                             stddev=Bob_test_sigma)
        Eve_code = Alice_code + \
            tf.random.normal(tf.shape(Alice_code), mean=0.0,
                             stddev=Eve_test_sigma)
        Bob_Predict = Bob_decoder.predict(Bob_code)
        Eve_Predict = Eve_decoder.predict(Eve_code)

        Bob_mse = np.sum(
            np.square(np.abs(np.subtract(Bob_Predict, TestData))))/TestSize/codeLen
        Eve_mse = np.sum(
            np.square(np.abs(np.subtract(Eve_Predict, TestData))))/TestSize/codeLen
        Bob_MSE[i] += Bob_mse
        Eve_MSE[i] += Eve_mse

        Bob_Data = np.where(Bob_Predict >= 0.5, 1, 0)
        Eve_Data = np.where(Eve_Predict >= 0.5, 1, 0)
        Bob_ser = np.sum(
            np.abs(np.subtract(Bob_Data, TestData)))/TestSize/codeLen
        Eve_ser = np.sum(
            np.abs(np.subtract(Eve_Data, TestData)))/TestSize/codeLen
        Bob_SER[i] += Bob_ser
        Eve_SER[i] += Eve_ser

        Bob_llr = Bob_Predict.flatten().astype(np.float64)
        for k in range(len(Bob_llr)):
            Bob_llr[k] = utils.llr(Bob_llr[k])
        Bob_Info = myldpc.decode(H, G, Bob_llr, Bob_test_snr, maxiter=1000)
        Eve_llr = Eve_Predict.flatten().astype(np.float64)
        for k in range(len(Bob_llr)):
            Eve_llr[k] = utils.llr(Eve_llr[k])
        Eve_Info = myldpc.decode(
            H, G, Eve_llr, Eve_test_snr, maxiter=1000)
        Bob_ber = np.sum(np.abs(np.subtract(Bob_Info, Info)))/TestSize/infoLen
        Eve_ber = np.sum(np.abs(np.subtract(Eve_Info, Info)))/TestSize/infoLen
        Bob_BER[i] += Bob_ber
        Eve_BER[i] += Eve_ber
        print(
            f'SNR: Bob: {Bob_test_snr}dB Eve: {Eve_test_snr}dB - Bob BER: {Bob_BER[i]:.4f} Eve BER: {Eve_BER[i]:.4f}')

MSE_fig = plt.figure(figsize=(10, 5))
plt.semilogy(Bob_snr_range, Bob_MSE, "o-")
plt.semilogy(Eve_snr_range, Eve_MSE, "s-")
plt.xlabel("SNR", fontsize=14)
plt.ylabel("MSE", fontsize=14, rotation=90)
plt.legend(['Bob_MSE', 'Eve_MSE'],
           prop={'size': 14}, loc='upper right')
plt.grid(True)
plt.show()
MSE_fig.savefig('python_autoencoder/MSE_2.png')

SER_fig = plt.figure(figsize=(10, 5))
plt.semilogy(Bob_snr_range, Bob_SER, "o-")
plt.semilogy(Eve_snr_range, Eve_SER, "s-")
plt.xlabel("SNR", fontsize=14)
plt.ylabel("SER", fontsize=14, rotation=90)
plt.legend(['Bob_SER', 'Eve_SER'],
           prop={'size': 14}, loc='upper right')
plt.grid(True)
plt.show()
SER_fig.savefig('python_autoencoder/SER_2.png')

BER_fig = plt.figure(figsize=(10, 5))
plt.semilogy(Bob_snr_range, Bob_BER, "o-")
plt.semilogy(Eve_snr_range, Eve_BER, "s-")
plt.xlabel("SNR", fontsize=14)
plt.ylabel("BER", fontsize=14, rotation=90)
plt.legend(['Bob_BER', 'Eve_BER'],
           prop={'size': 14}, loc='upper right')
plt.grid(True)
plt.show()
BER_fig.savefig('python_autoencoder/BER_2.png')
