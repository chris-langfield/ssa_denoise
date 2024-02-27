import numpy as np
import scipy

def denoise_wf(w, r, freq_range, fs=30_000, traj_shape=None):
    """
    :param w: (ntr, ns) Waveform array (real-space)
    :param freq_range: tuple (min_frequency, max_frequency) in Hz.
    :param r: Derank to this value (int)
    :param fs: Sampling rate (Hz) default: 30,000
    :param traj_shape: Shape of trajectory matrix (default: ntr//2 x ntr-ntr//2)
    """
    # realspace waveform dimensions
    ntr, ns = w.shape

    # memory of input array
    imem = np.nbytes[w.dtype] * ntr * ns
    
    # trajectory matrix dimensions
    if not traj_shape:
        L = int(ntr // 2)
        K = int(ntr - L)
    else:
        L, K = traj_shape

    # min and max frequency
    min_freq, max_freq = freq_range

    # Get FFT and frequencies
    W = np.fft.rfft(w)

    freqs = np.fft.rfftfreq(ns, d=1/fs)
    nfreq = freqs.shape[0]
    min_f = np.searchsorted(freqs, min_freq) # idx of min freq
    max_f = np.searchsorted(freqs, max_freq) #
    nfreq_use = max_f - min_f

    # construct sequence of Hankel trajectory matrices for each frequency slice
    hanks = np.stack([hank_mat(W, f, L, K) for f in range(min_f, max_f)])

    # perform deranking of each frequency slice
    # save SVD info for each frequency to output
    deranked_hanks = np.zeros(hanks.shape, np.complex64)
    svd_U = np.zeros((nfreq_use, r, r), np.complex64)
    svd_V = np.zeros((nfreq_use, r, r), np.complex64)
    svd_S = np.zeros((nfreq_use, r,), np.complex64)
    for f in range(max_f-min_f):
        deranked_hank, u, s, v = derank(hanks[f, :, :], r)
        deranked_hanks[f, :, :] = deranked_hank
        svd_U[f, :, :] = u[:r, :r]
        svd_V[f, :, :] = v[:r, :r]
        svd_S[f, :] = s[:r]

    # reconstruct wf
    W_ssa = np.stack([diagonal_average(deranked_hanks[i]) for i in range(nfreq_use)] + [np.array([0]*(ntr-1)) for l in range(nfreq-nfreq_use)]).T
    w_ssa = np.fft.irfft(W_ssa)

    # SVD "basis"
    elementary_matrices = get_elementary_matrices(hanks, r)

    # compression
    omem = np.nbytes[np.complex64] * (2*np.size(svd_U) + np.size(svd_S))

    # print(f"imem: {imem} bytes")
    # print(f"omem: {omem} bytes")

    return w_ssa, svd_U, svd_S, svd_V, elementary_matrices

def get_elementary_matrices(hanks, r):
    """
    Given a stack of hankel matrices stacked by frequency return
    the stack of elementary matrix cubes.
    :param hanks: Stack of hanks
    :param r: rank
    """
    nf = hanks.shape[0]
    svd_basis = np.zeros((nf, r, r, r), np.complex64)
    for f in range(nf):
        U, S, V = np.linalg.svd(hanks[f, :, :])
        svd_basis[f, :, :, :]= np.array([S[i]* np.outer(U[:,i][:r], V.T[:,i][:r]) for i in range(r)])
    return svd_basis

def hank_column(W, i, f, L):
    """
    Returns one trajectory from an fx space waveform at a given
    frequency slice
    :param W: Waveform array (ntr, nfreq) in fx space
    :param i: Trajectory number
    :param f: frequency slice (INDEX not value)
    :param L: Window size of trajectory matrix
    """
    return W[i:i + L, f]

def hank_mat(W, f, L, K):
    """
    For a given frequency index, return the hankel trajectory matrix of the traces.
    :param W: Waveform array (ntr, nfreq) in fx space
    :param f: frequency slice index
    :param L: Row size of trajectory matrix
    :param K: Column size of trajectory matrix
    """
    # eq (4) and (9)
    return np.column_stack([hank_column(W, i, f, L) for i in range(K)])[::-1]


def diagonal_average(X):
    """
    :param X: mxn Numpy array
    :return: A 1-D array of length (m+n-1) containing the i'th diagonal averages. 
    """
    X_rev = X
    return np.array([X_rev.diagonal(p).mean() for p in range(-X.shape[0]+1, X.shape[1])])

def derank(T, r):
    u, s, v = np.linalg.svd(T)
    T_ = np.zeros_like(T)
    for i in np.arange(r):
        T_ += s[i] * np.outer(u.T[i], v[i])
    return T_, u, s, v

def reconstruct_from_svd(svd_u, svd_s, svd_v, nfreq_use, nfreq=61, traj_shape=None):
    """
    Given truncated SVD components, reconstruct the WF.

    svd_u, svd_s, svd_v are outputs from denoise_wf.
    """
    if traj_shape is None:
        traj_shape = (20, 20)
    L, K = traj_shape
    nf, r = svd_s.shape
    hank_stack = np.zeros((nf, L, L), np.complex64)
    for f in range(nf):
        T_ = np.zeros((L, L), np.complex64)
        for i in range(r):
            T_[:r, :r] += svd_s[f,:][i] * np.outer(svd_u[f,:,:].T[i], svd_v[f,:,:][i])
        hank_stack[f, :, :] = T_
    W =np.stack([diagonal_average(hank_stack[i]) for i in range(nfreq_use)] + [np.array([0]*(ntr-1)) for l in range(nfreq-nfreq_use)]).T
    return np.fft.irfft(W)
    
def flattop_gauss(shape, sx, sy, px, py, shift=(0,0)):
    """
    :param shape:
    :param px: power in x
    :param py: power in y
    :param sx: stdev in x
    :param sy: stdev in y
    :param shift: xy shift (tuple)
    """
    xlen, ylen = shape
    xshift, yshift = shift
    x, y = np.meshgrid(np.arange(-xlen//2+1, xlen//2+1), np.arange(-ylen//2, ylen//2))
    gauss = np.exp(-np.power(x - xshift, px)/(2*sx**2) 
                   -np.power(y - yshift, py)/(2*sy**2)
    )
    return gauss

    



