import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy import special
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


M = 16
M_sec = 4
k = int(np.log2(M))
n = 24
TRAINING_SNR = 10  # snr = ebno * k/n
SAMPLE_SIZE = 5000
messages = np.random.randint(M, size=SAMPLE_SIZE)

# %%
one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(M)])
data_oneH = one_hot_encoder.fit_transform(messages.reshape(-1, 1))

# Generate Training Data
# x = tf.random.uniform(shape=[SAMPLE_SIZE], minval=0, maxval=M, dtype=tf.int64)
# x_1h = tf.one_hot(x, M)
# dataset = tf.data.Dataset.from_tensor_slices(x_1h)

# %%


def snr_to_noise(snrdb):
    '''Transform snr to noise power'''
    snr = 10**(snrdb/10)
    noise_std = 1/np.sqrt(2*snr)  # 1/np.sqrt(2*(k/n)*ebno) for ebno to noise
    return noise_std


# %%
noise_std = snr_to_noise(TRAINING_SNR)
noise_std_eve = snr_to_noise(7)

# custom functions / layers without weights
norm_layer = keras.layers.Lambda(lambda x: tf.divide(
    x, tf.sqrt(2*tf.reduce_mean(tf.square(x)))))
shape_layer = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2, n]))
shape_layer2 = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2*n]))
channel_layer = keras.layers.Lambda(lambda x:
                                    tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std)))
channel_layer_eve = keras.layers.Lambda(lambda x:
                                        tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std_eve)))

encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[M]),
    keras.layers.Dense(M, activation="elu"),
    keras.layers.Dense(2*n, activation=None),
    shape_layer,
    norm_layer])


def init_kmeans(symM=16, satellites=4, n=100):
    '''Initializes equal sized clusters with the whole message set'''
    inp = np.eye(symM, dtype=int)
    unit_codewords = encoder.predict(inp)
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


# Initlizing kmeans for the security procedure
kmeans = init_kmeans(M, M_sec, n)
generator_matrix = generate_mat(kmeans.labels_, M_sec, M)
