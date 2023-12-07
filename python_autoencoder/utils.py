import numpy as np


def snr_to_noise(snrdb):
    snr = 10 ** (snrdb / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


def llr(x):
    if x > 1-1e-8:
        x = 1-1e-8
    elif x < 1e-8:
        x = 1e-8
    return np.log((1-x)/x)


def int2bitarray(n, k):
    """Change an array's base from int (base 10) to binary (base 2)."""
    binary_string = bin(n)
    length = len(binary_string)
    bitarray = np.zeros(k, 'int')
    for i in range(length - 2):
        bitarray[k - i - 1] = int(binary_string[length - i - 1])

    return bitarray


def bitarray2int(bitarray):
    """Change array's base from binary (base 2) to int (base 10)."""
    bitstring = "".join([str(i) for i in bitarray])

    return int(bitstring, 2)
