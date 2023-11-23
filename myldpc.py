import pyldpc
import numpy as np
import warnings
from pyldpc import utils
from numba import njit, int64, types, float64

def ldpcQC(blockSize,baseGraph):
    mb,nb = baseGraph.shape
    m=mb*blockSize
    n=nb*blockSize
    I=np.eye(blockSize)
    H=np.zeros((m,n),dtype=int)
    for i in range(mb):
        for j in range(nb):
            if baseGraph[i,j]>=0:
                H[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize]=np.roll(I,baseGraph[i,j],axis=1)
    return H

def encode(G,data):
    x = pyldpc.utils.binaryproduct(G, data)
    return x

def decode(H,G,code,snr_db,maxiter=1000):
    decoded=decode_internal(H,code,snr_db,maxiter)
    info=get_message(G,decoded)
    return info

def decode_internal(H,y,snr,maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Decoding is performed in parallel if multiple codewords are passed in y.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code, n_messages) or (n_code,). Received message(s) in the
        codeword space.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    x: array (n_code,) or (n_code, n_messages) the solutions in the
        codeword space.

    """
    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

    solver = _logbp_numba

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # step 0: initialization

    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages))

    Lr = np.zeros(shape=(m, n, n_messages))
    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                      nodes_values, Lc, Lq, Lr, n_iter)
        x = np.array(L_posteriori <= 0).astype(int)
        product = utils.incode(H, x)
        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()


output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
                               float64[:, :]))


@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :],  float64[:, :, :], int64), cache=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
                 n_iter):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal

    bits_counter = 0
    nodes_counter = 0
    for i in range(m):
        # ni = bits[i]
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        # mj = nodes[j]
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


@njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
                       float64[:, :], float64[:, :, :],  float64[:, :, :],
                       int64), cache=True)
def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc,
                         Lq, Lr, n_iter):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal
    for i in range(m):
        ni = bits_values[i]
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        mj = nodes_values[j]
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    for j in range(n):
        mj = nodes_values[j]
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    rtG, rx = utils.gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= utils.binaryproduct(rtG[i, list(range(i+1, k))],
                                          message[list(range(i+1, k))])

    return abs(message)

# baseGraph=np.mat([[0, -1,-1,-1, 0, 0 ,-1 ,-1 ,0 ,-1 ,-1, 0 ,1 ,0 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1, -1],
#     [22, 0 ,-1, -1, 17, -1, 0, 0 ,12 ,-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
#     [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
#     [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
#     [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
#     [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
#     [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
#     [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, 0, 0, -1, -1],
#     [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
#     [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
#     [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])