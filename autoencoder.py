from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def init_autoencoder(Snr,innerLen,outerLen,learning_rate=0.01):
    #初始化自动编码器
    encoder_input = layers.InputLayer(input_shape=(innerLen,))
    encoder_output1 = layers.Dense(innerLen)
    encoder_output2 = layers.ReLU()
    encoder_output3 = layers.BatchNormalization()
    encoder_output4 = layers.Dense(outerLen)
    encoder_output5 = layers.ReLU()
    encoder_output6 = layers.BatchNormalization()
    encoder=keras.models.Sequential([encoder_input,encoder_output1,encoder_output2,encoder_output3,
                                     encoder_output4,encoder_output5,encoder_output6,encoder_output6])
    #信道噪声
    channel=layers.GaussianNoise(Snr)
   
    # 解码器
    decoder_input = layers.InputLayer(input_shape=(outerLen,))
    decoder_output1 = layers.Dense(outerLen)
    decoder_output2 = layers.ReLU()
    decoder_output3 = layers.BatchNormalization()
    decoder_output4 = layers.Dense(innerLen)
    decoder_output5 = layers.ReLU()
    decoder_output6 = layers.BatchNormalization()
    decoder_output7 = layers.Dense(innerLen, activation="sigmoid")
    
    decoder=keras.models.Sequential([decoder_input,decoder_output1,decoder_output2,decoder_output3,
                                      decoder_output4,decoder_output5,decoder_output6,decoder_output7])
       
    # 构建自动编码器模型
    autoencoder = keras.models.Sequential([encoder, channel, decoder])
     
    # 定义损失函数和优化器
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate)

    # 编译自动编码器
    autoencoder.compile(optimizer=optimizer, loss=loss_fn,metrics=['accuracy'])
       
    return autoencoder

def train_autoencoder(nEpochs,BatchSize,learning_rate,lr_scheduler,SampleData,autoencoder):
    # 训练自动编码器
    avg_losses=np.zeros(nEpochs)
    for epoch in range(nEpochs):
        # 在每个epoch开始时初始化损失值
        epoch_loss = 0.0
        # 更新学习率
        learning_rate = lr_scheduler(epoch,learning_rate)
        keras.backend.set_value(autoencoder.optimizer.learning_rate, learning_rate)

        # 遍历训练数据集的批次
        for step in range(0, len(SampleData), BatchSize):
            # 获取当前批次的输入数据
            x_batch = SampleData[step:step+BatchSize]
            
            # 计算当前批次的损失值
            loss = autoencoder.train_on_batch(x_batch, x_batch)
            
            # 累加当前批次的损失值到epoch_loss
            epoch_loss += loss
        
        # 计算平均损失值
        avg_loss = epoch_loss / (len(SampleData) // BatchSize)
        # 记录当前epoch的平均损失值
        avg_losses[epoch] = avg_loss 
        
        # 输出当前epoch的信息
        print(f'Epoch {epoch+1}/{nEpochs} - Loss: {avg_loss:.4f}')
    return avg_losses

def show_losses(nEpochs,avg_losses):
    fig = plt.figure(figsize=(18, 18))
    plt.plot(nEpochs, avg_losses, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.gca().set_ylim(-2, 2)
    plt.gca().set_xlim(-2, 2)
    plt.show()
    return fig