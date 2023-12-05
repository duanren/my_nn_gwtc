import utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 加载已训练模型
Alice_encoder_model_path_1 = "Alice_encoder_1.keras"
Bob_decoder_model_path_1 = "Bob_decoder_1.keras"
Eve_decoder_model_path_1 = "Eve_decoder_1.keras"

Alice_encoder = keras.models.load_model(Alice_encoder_model_path_1)
Bob_decoder = keras.models.load_model(Bob_decoder_model_path_1)
Eve_decoder = keras.models.load_model(Eve_decoder_model_path_1)

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


def init_kmeans(symM=16, satellites=4, n=100):
    '''Initializes equal sized clusters with the whole message set'''
    inp = np.eye(symM, dtype=int)
    unit_codewords = Alice_encoder.predict(inp)
    kmeans = KMeans(n_clusters=satellites)
    X = unit_codewords.reshape(symM, 2 * n)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]
                              ).repeat(satellites, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1] // satellites
    kmeans.labels_ = clusters
    return kmeans


def generate_mat(kmeans_labels, satellites=4, symM=16):
    '''Generates the matrix for equalization of the input distribution on Eves side'''
    gen_matrix = np.zeros((symM, symM))
    for j in range(satellites):
        for i in range(symM):
            if kmeans_labels[i] == j:
                for k in range(symM):
                    if kmeans_labels[k] == j:
                        gen_matrix[i, k] = 1/satellites
    gen_mat = tf.cast(gen_matrix, tf.float64)
    return gen_mat


# 初始化k聚类
kmeans = init_kmeans(innerLen, modLen, innerLen)
generator_matrix = generate_mat(kmeans.labels_, modLen, innerLen)

# 物理层安全训练
# 生成训练集
SampleSize = 102400
SampleData = np.random.randint(0, 2, (SampleSize, innerLen))

# 训练
nEpochs = 250
BatchSize = 120
alpha = 0.7
print('autoencoder_2 training...')
# 训练自动编码器
Bob_avg_losses = np.zeros(nEpochs)
Eve_avg_losses = np.zeros(nEpochs)
Sec_avg_losses = np.zeros(nEpochs)
for epoch in range(nEpochs):
    # 在每个epoch开始时初始化损失值
    Bob_epoch_loss = 0.0
    Eve_epoch_loss = 0.0
    Sec_epoch_loss = 0.0
    # 更新学习率
    Bob_learning_rate = keras.backend.get_value(
        Bob_autoencoder.optimizer.learning_rate)
    Bob_learning_rate = lr_scheduler(epoch, Bob_learning_rate)
    keras.backend.set_value(
        Bob_autoencoder.optimizer.learning_rate, Bob_learning_rate)

    # 遍历训练数据集的批次
    for step in range(0, len(SampleData), BatchSize):
        # 获取当前批次的输入数据
        X_batch_Bob = SampleData[step:step+BatchSize]
        X_batch_Bob = X_batch_Bob.reshape(-1, innerLen)
        X_batch_Eve = tf.matmul(X_batch_Bob, generator_matrix)
        with tf.GradientTape() as tape:
            Bob_predict = Bob_autoencoder(X_batch_Bob, training=True)
            Eve_predict = Eve_autoencoder(X_batch_Bob, training=False)

            Bob_loss = tf.reduce_mean(loss_fn(X_batch_Bob, Bob_predict))
            Eve_loss = tf.reduce_mean(loss_fn(X_batch_Eve, Eve_predict))
            Sec_loss = alpha*Bob_loss + (1-alpha)*Eve_loss

            Bob_epoch_loss += Bob_loss
            Eve_epoch_loss += Eve_loss
            Sec_epoch_loss += Sec_loss

            gradients = tape.gradient(
                Sec_loss, Bob_autoencoder.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, Bob_autoencoder.trainable_variables))

    # 计算平均损失值
    Bob_avg_loss = Bob_epoch_loss / (len(SampleData) // BatchSize)
    Eve_avg_loss = Eve_epoch_loss / (len(SampleData) // BatchSize)
    Sec_avg_loss = Eve_epoch_loss / (len(SampleData) // BatchSize)
    # 记录当前epoch的平均损失值
    Bob_avg_losses[epoch] = Bob_avg_loss
    Eve_avg_losses[epoch] = Eve_avg_loss
    Sec_avg_losses[epoch] = Eve_avg_loss

    # 输出当前epoch的信息
    print(
        f'Epoch {epoch+1}/{nEpochs} - Bob Loss: {Bob_avg_loss:.4f} Eve Loss: {Eve_avg_loss:.4f} Sec Loss: {Sec_avg_loss:.4f}')

print('autoencoder_2 is trained.')
# 保存模型
Alice_encoder_model_path_2 = "Alice_encoder_2.keras"
Bob_decoder_model_path_2 = "Bob_decoder_2.keras"
Eve_decoder_model_path_2 = "Eve_decoder_2.keras"

Alice_encoder.save(
    filepath=Alice_encoder_model_path_2, save_format='keras')
Bob_decoder.save(
    filepath=Bob_decoder_model_path_2, save_format='keras')
Eve_decoder.save(
    filepath=Eve_decoder_model_path_2, save_format='keras')

# 显示损失函数
loss_fig = plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, nEpochs+1), Bob_avg_losses, "o-")
plt.plot(np.arange(1, nEpochs+1), Eve_avg_losses, "s-")
plt.plot(np.arange(1, nEpochs+1), Sec_avg_losses, "b-")
plt.xlabel("nEpochs", fontsize=14)
plt.ylabel("avg_losses", fontsize=14, rotation=90)
plt.legend(['Bob_avg_losses', 'Eve_avg_losses', 'Sec_avg_losses'],
           prop={'size': 14}, loc='upper right')
plt.grid(True)
plt.show()

# 保存损失函数图像
loss_fig.savefig('loss_2.png')
