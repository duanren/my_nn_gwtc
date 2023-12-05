import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

import utils

Alice_encoder_model_path_1 = "Alice_encoder_1.keras"
Bob_decoder_model_path_1 = "Bob_decoder_1.keras"
Eve_decoder_model_path_1 = "Eve_decoder_1.keras"

# 初始化Bob和Eve的snr
Bob_train_snr = 10
Eve_train_snr = 3
Bob_train_sigma = utils.snr_to_noise(Bob_train_snr)
Eve_train_sigma = utils.snr_to_noise(Eve_train_snr)
Bob_channel = layers.Lambda(lambda x:
                            tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=Bob_train_sigma)))
Eve_channel = layers.Lambda(lambda x:
                            tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=Eve_train_sigma)))
# 初始化模型参数
innerLen = 24
modLen = 4
outerLen = innerLen * 2 // modLen


encoder_input = layers.InputLayer(input_shape=(innerLen,))
encoder_output1 = layers.Dense(innerLen * 2)
encoder_output2 = layers.ReLU()
encoder_output3 = layers.BatchNormalization()
encoder_output4 = layers.Dense(outerLen)
encoder_output5 = layers.ReLU()
encoder_output6 = layers.BatchNormalization()
Alice_encoder = keras.models.Sequential([encoder_input, encoder_output1, encoder_output2,
                                         encoder_output3, encoder_output4, encoder_output5, encoder_output6])


decoder_input = layers.InputLayer(input_shape=(outerLen,))
decoder_output1 = layers.Dense(innerLen * 2)
decoder_output2 = layers.ReLU()
decoder_output3 = layers.BatchNormalization()
decoder_output4 = layers.Dense(innerLen)
decoder_output5 = layers.ReLU()
decoder_output6 = layers.BatchNormalization()
decoder_output7 = layers.Dense(innerLen, activation="sigmoid")
Bob_decoder = keras.models.Sequential([decoder_input, decoder_output1, decoder_output2,
                                       decoder_output3, decoder_output4, decoder_output5, decoder_output6, decoder_output7])


decoder_input = layers.InputLayer(input_shape=(outerLen,))
decoder_output1 = layers.Dense(outerLen)
decoder_output2 = layers.ReLU()
decoder_output3 = layers.BatchNormalization()
decoder_output4 = layers.Dense(innerLen)
decoder_output5 = layers.ReLU()
decoder_output6 = layers.BatchNormalization()
decoder_output7 = layers.Dense(innerLen, activation="sigmoid")
Eve_decoder = keras.models.Sequential([decoder_input, decoder_output1, decoder_output2,
                                       decoder_output3, decoder_output4, decoder_output5, decoder_output6, decoder_output7])

# 构建自动编码器模型
Bob_autoencoder = keras.models.Sequential(
    [Alice_encoder, Bob_channel, Bob_decoder])
Eve_autoencoder = keras.models.Sequential(
    [Alice_encoder, Eve_channel, Eve_decoder])

# 定义学习率调度器


def lr_scheduler(epoch, lr):
    if (epoch + 1) % 50 == 0:
        return lr * 0.5  # 每50个epoch学习率减半
    else:
        return lr


# 定义学习率调度器回调
lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

# 定义损失函数和优化器
learning_rate = 0.1
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.legacy.Adam(learning_rate)

# 编译自动编码器
Bob_autoencoder.compile(optimizer=optimizer, loss=loss_fn)
Eve_autoencoder.compile(optimizer=optimizer, loss=loss_fn)

print('autoencoder init done.')


# 第一轮训练
# 生成训练集
SampleSize = 102400
SampleData = np.random.randint(0, 2, (SampleSize, innerLen))

# 训练
nEpochs = 500
BatchSize = 120

print('autoencoder_1 training...')
# 训练自动编码器
Bob_avg_losses = np.zeros(nEpochs)
Eve_avg_losses = np.zeros(nEpochs)
for epoch in range(nEpochs):
    # 在每个epoch开始时初始化损失值
    Bob_epoch_loss = 0.0
    Eve_epoch_loss = 0.0
    # 更新学习率
    Bob_learning_rate = keras.backend.get_value(
        Bob_autoencoder.optimizer.learning_rate)
    Bob_learning_rate = lr_scheduler(epoch, Bob_learning_rate)
    keras.backend.set_value(
        Bob_autoencoder.optimizer.learning_rate, Bob_learning_rate)
    Eve_learning_rate = keras.backend.get_value(
        Eve_autoencoder.optimizer.learning_rate)
    Eve_learning_rate = lr_scheduler(epoch, Eve_learning_rate)
    keras.backend.set_value(
        Eve_autoencoder.optimizer.learning_rate, Eve_learning_rate)

    # 遍历训练数据集的批次
    for step in range(0, len(SampleData), BatchSize):
        # 获取当前批次的输入数据
        x_batch = SampleData[step:step+BatchSize]

        # 计算当前批次的损失值
        Bob_loss = Bob_autoencoder.train_on_batch(x_batch, x_batch)
        Eve_loss = Eve_autoencoder.train_on_batch(x_batch, x_batch)

        # 累加当前批次的损失值到epoch_loss
        Bob_epoch_loss += Bob_loss
        Eve_epoch_loss += Eve_loss

    # 计算平均损失值
    Bob_avg_loss = Bob_epoch_loss / (len(SampleData) // BatchSize)
    Eve_avg_loss = Eve_epoch_loss / (len(SampleData) // BatchSize)
    # 记录当前epoch的平均损失值
    Bob_avg_losses[epoch] = Bob_avg_loss
    Eve_avg_losses[epoch] = Eve_avg_loss

    # 输出当前epoch的信息
    print(
        f'Epoch {epoch+1}/{nEpochs} - Bob Loss: {Bob_avg_loss:.4f} Eve Loss: {Eve_avg_loss:.4f}')

print('autoencoder_1 is trained.')
# 保存模型
Alice_encoder.save(
    filepath=Alice_encoder_model_path_1, save_format='keras')
Bob_decoder.save(
    filepath=Bob_decoder_model_path_1, save_format='keras')
Eve_decoder.save(
    filepath=Eve_decoder_model_path_1, save_format='keras')
# 显示损失函数
loss_fig = plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, nEpochs+1), Bob_avg_losses, "o-")
plt.plot(np.arange(1, nEpochs+1), Eve_avg_losses, "s-")
plt.xlabel("nEpochs", fontsize=14)
plt.ylabel("avg_losses", fontsize=14, rotation=90)
plt.legend(['Bob_avg_losses', 'Eve_avg_losses'],
           prop={'size': 14}, loc='upper right')
plt.grid(True)
plt.show()

# 保存损失函数图像
loss_fig.savefig('loss_1.png')
