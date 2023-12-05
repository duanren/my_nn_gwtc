import numpy as np

def snr_to_noise(snrdb):
    snr = 10 ** (snrdb / 10)
    noise_std = 1 / np.sqrt(2 * snr)  
    return noise_std

def llr(x):
    if x>1-1e-8:
        x=1-1e-8
    elif x<1e-8:
        x=1e-8
    return np.log((1-x)/x)

