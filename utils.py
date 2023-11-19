import numpy as np

def snr_to_noise(snrdb):
    snr = 10 ** (snrdb / 10)
    noise_std = 1 / np.sqrt(2 * snr)  
    return noise_std

def llr(p):
    if p>=1:
        p=0.999999999999999
    elif p<=0:
        p=0.000000000000001
    return np.log((1 - p) / p)

def llr_array(p_array):
    l=p_array
    for i in range(len(l)):
        l[i]=llr(l[i])
    return l

def hard_decide(x):
    if x>=0.5:
        return 1
    elif x<0.5:
        return 0

def hard_decide_array(x_array):
    y=x_array
    for i in range(len(y)):
        y[i]=hard_decide(y[i])
    return y
