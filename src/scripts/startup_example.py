import numpy as np
from pathlib import Path
from ssa_denoise import *
import matplotlib.pyplot as plt
from tqdm import trange


def trace_plot(arr, ax, channel_range=None, spike_range=None, title=""):
    if spike_range is None:
        spike_range = np.arange(0, 121)
    if channel_range is None:
        channel_range = np.arange(0, 384)
    ax.set_facecolor("#222")
    all_norms = np.linalg.norm(np.abs(arr[spike_range,:]), axis=0)
    max_norm = np.max(all_norms)
    all_norms = np.sqrt(all_norms / max_norm) / 2.
    for i in channel_range:
        y = i + arr[spike_range, i]
        ax.plot(spike_range, y, 
                color = 
                (0.4 + all_norms[i], 0.3 + 0.4*all_norms[i], 0.3 + all_norms[i], .7))
    ax.set_title(title)

data_path = Path("/Users/chris/Downloads")
raw_waveforms = np.load(data_path.joinpath("benchmark_wfs.npy"))

bump4 = flattop_gauss((121, 40), 1000, 200, 4, 4, shift=(-15,0))
bump2 = flattop_gauss((121, 40), 32, 20, 2, 2, shift=(-20,0))
bump0 = np.ones((40, 121))
r = 5
freq_range = (0, 8000)

for i in trange(10000):
    raw = raw_waveforms[i, :, :].T
    arr, svd_u, svd_s, svd_v, e = denoise_wf(raw * bump0, r, freq_range)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(raw, cmap="gray")
    ax[0,1].imshow(arr, cmap="gray")
    trace_plot(np.flipud(raw).T *5e4, ax[1,0], np.arange(40), np.arange(121))
    trace_plot(np.flipud(arr).T *5e4, ax[1,1], np.arange(39), np.arange(120))
    fig.suptitle(f"Rank: {r}, Freq: {freq_range[0]}-{freq_range[1]} Hz")
    #fig.savefig(f"1a27_c{i}_r{r}_f_{freq_range}.png")

def reconstruct_from_svd_clip(svd_u, svd_s, svd_v):
    """
    Given truncated SVD components, reconstruct the WF.

    svd_u, svd_s, svd_v are outputs from denoise_wf.
    """
    nf, r = svd_s.shape
    hank_stack = np.zeros((nf, r, r), np.complex64)
    for f in range(nf):
        T_ = np.zeros((r, r), np.complex64)
        for i in range(r):
            T_[:r, :r] += svd_s[f,:][i] * np.outer(svd_u[f,:,:].T[i], svd_v[f,:,:][i])
        hank_stack[f, :, :] = T_
    W =np.stack([diagonal_average(hank_stack[i]) for i in range(nf)])# + [np.array([0]*(ntr-1)) for l in range(nfreq-nfreq_use)]).T
    return np.fft.irfft(W)


freqs = np.fft.rfftfreq(121, d=1/30_000)
freq_idx = np.searchsorted(freqs, [freq_range[0], freq_range[1]])
freqs = freqs.astype(int)
for i in trange(130):
    raw = raw_waveforms[i*500, :, :].T
    arr, svd_u, svd_s, svd_v, e = denoise_wf(raw * bump0, r, freq_range)
    # plot singular values
    fig, ax = plt.subplots()
    ax.set_title("Singular values")
    ax.imshow(np.real(svd_s.T))
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Singular value")
    xtick_loc = np.arange(0, 31, 5)
    labs = list(freqs[slice(*freq_idx)][xtick_loc])
    ax.set_xticklabels([0] + labs)
    ax.set_yticks(np.arange(0, 5))

svd_vecs = np.zeros((33*r, 5000), np.float32)
for i in trange(5000):
    raw = raw_waveforms[i, :, :].T
    arr, svd_u, svd_s, svd_v, e = denoise_wf(raw * bump0, r, freq_range)
    svd_vecs[:, i] = svd_s.flatten().astype(np.float32)

realspace_trunc = np.zeros((33*16, 5000), np.float32)
for i in trange(6, 9):
    raw = raw_waveforms[500*i, :, :].T
    arr, svd_u, svd_s, svd_v, e = denoise_wf(raw * bump0, r, freq_range)
    plt.matshow(reconstruct_from_svd_clip(svd_u, svd_s, svd_v).astype(np.float32).T, cmap="gray")

from scipy.spatial import distance_matrix
n_units = 10
n_spikes = 5000
n_samples = 33*r

E = distance_matrix(realspace_trunc.T, realspace_trunc.T)
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_units)
kmeans.fit_predict(E)

order = np.argsort(kmeans.labels_)

plt.matshow(E[order,:])

ground_truth_labels =  np.array([[i]*500 for i in range(10)]).flatten()


sh_TSNE = TSNE(n_components=3).fit_transform(E)

fig, axs = plt.subplots(figsize=(8, 8))
axs.scatter(sh_TSNE[:, 0], sh_TSNE[:, 1], c=ground_truth_labels)

# 3d 

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(sh_TSNE[:, 0], sh_TSNE[:, 1], sh_TSNE[:, 2], c=ground_truth_labels)

import umap
import umap.plot
mapper = umap.UMAP().fit(E)

umap.plot.points(mapper, labels=kmeans.labels_)

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix

adjusted_rand_score(ground_truth_labels, kmeans.labels_)
adjusted_mutual_info_score(ground_truth_labels, kmeans.labels_)
conting = contingency_matrix(ground_truth_labels, kmeans.labels_)


# generate the stack of hankel traj matrices
def hank_stack(w):
    ntr, ns = w.shape
    L = int(ntr // 2)
    K = int(ntr - L)

    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(ns, d=1/30_000)

    hanks = np.stack([hank_mat(W, f, L, K) for f in range(0, len(freqs))])
    return hanks

i = 10
raw = raw_waveforms[i,:,:].T
arr, _, _, _, _ = denoise_wf(raw, 5, [0, 8000])

hanks = np.abs(hank_stack(raw))
fig, ax = plt.subplots()
ax.set_yticks([])
ax.set_xticks([])
ax.imshow(raw, cmap="gray")
ax.imshow(arr, cmap="gray")
ax.imshow(np.abs(np.fft.rfft(raw)), cmap="gray")

fig, ax = plt.subplots()
trace_plot(np.flipud(raw).T *5e4, ax, np.arange(40), np.arange(121))
fig, ax = plt.subplots()
trace_plot(np.flipud(arr).T *5e4, ax, np.arange(39), np.arange(120))

freqs = np.fft.rfftfreq(121, d=1/30_000).astype(int)
fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i, j].matshow(hanks[4*i + j % 4], cmap="gray")
        ax[i, j].set_yticks([])
        ax[i, j].set_xticks([])
        ax[i, j].set_title(f"{freqs[4*i + j %4]} Hz")
fig.tight_layout()
