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
r = 5
freq_range = (0, 8000)
for i in trange(10):
    raw = raw_waveforms[i, :, :].T
    arr, svd_u, svd_s, svd_v, e = denoise_wf(raw * bump4, r, freq_range)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(raw, cmap="gray")
    ax[0,1].imshow(arr, cmap="gray")
    trace_plot(np.flipud(raw).T *5e4, ax[1,0], np.arange(40), np.arange(121))
    trace_plot(np.flipud(arr).T *5e4, ax[1,1], np.arange(39), np.arange(120))
    fig.suptitle(f"Rank: {r}, Freq: {freq_range[0]}-{freq_range[1]} Hz")
    fig.savefig(f"1a27_c{i}_r{r}_f_{freq_range}.png")


