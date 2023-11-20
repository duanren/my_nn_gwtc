import numpy as np

def snr_to_noise(snrdb):
    snr = 10 ** (snrdb / 10)
    noise_std = 1 / np.sqrt(2 * snr)  
    return noise_std



