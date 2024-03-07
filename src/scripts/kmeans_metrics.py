from tqdm import trange
import numpy as np
from pathlib import Path
from ssa_denoise import *
import umap
import umap.plot
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix
import matplotlib.pyplot as plt

class WaveformData:

    def __init__(self, path, cluster_ids):
        """
        Assumes a (nwf, ns, nc) npy array
        cluster_ids should have shape (nwf,)
        """
        self.path = path
        self.cluster_ids = cluster_ids
        self.waveforms = np.load(path)

    def __getitem__(self, k):
        return self.waveforms[k]
    
    def from_cluster(self, cluster_id):
        idx = self.cluster_ids == cluster_id
        return self[idx, :, :]

    def from_clusters(self, cluster_id_arr):
        idx = np.in1d(self.cluster_ids, cluster_id_arr)
        return self[idx, :, :]

    @property
    def shape(self):
        return self.waveforms.shape
    

data_path = Path("/Users/chris/Downloads")
cluster_ids = np.array([[i]*500 for i in range(130)]).flatten()
wfs = WaveformData(data_path.joinpath("benchmark_wfs.npy"),
                   cluster_ids=cluster_ids)

class Fingerprinter:

    def __init__(self, waveforms, func, cluster_ids, rs, freqs, distance_metric="euclidean"):
        self.waveforms = waveforms
        self.func = func
        self.cluster_ids = cluster_ids
        self.rs = rs
        self.freqs = freqs
        self.distance_metric = distance_metric
        self.nclusters = len(cluster_ids)
        print(f"Number of clusters: {len(cluster_ids)}")
        self.nwf = sum(np.in1d(self.waveforms.cluster_ids, cluster_ids))
        print(f"Number of waveforms: {self.nwf}")

    def ground_truth_labels(self):
        return self.waveforms.cluster_ids[np.in1d(self.waveforms.cluster_ids, self.cluster_ids)]
    
    def compute_features(self, r, max_freq):
        # get shape of output
        wfs = self.waveforms.from_clusters(self.cluster_ids)

        _, svd_u, svd_s, svd_v, _ = denoise_wf(wfs[0, :, :].T, r, [0, max_freq])
        _feat = self.func(svd_u, svd_s, svd_v)
        feat_size = _feat.size
        feats = np.zeros((self.nwf,) + (feat_size,), np.float32)
        for i in trange(self.nwf):
            _, svd_u, svd_s, svd_v, _ = denoise_wf(wfs[i,:,:].T, r, [0, max_freq])
            feats[i, :] = self.func(svd_u, svd_s, svd_v).flatten().astype(np.float32)
        return feats
    
    def compute_distance_matrix(self, feats):
        return distance_matrix(feats, feats)
    
    def kmeans(self, distance_matrix):
        kmeans = KMeans(n_clusters = self.nclusters)
        kmeans.fit_predict(distance_matrix)
        return kmeans.labels_
    
    def grid_plots(self, cluster_plots=True):

        nr = len(self.rs)
        nf = len(self.freqs)

        if cluster_plots:
            tsne_fig, tsne_ax = plt.subplots(nr, nf)
            umap_fig, umap_ax = plt.subplots(nr, nf)

        met_fig, met_ax = plt.subplots(2, 2)
        ground_truth_labels = self.ground_truth_labels()

        rand_scores = np.zeros((nr, nf), np.float32)
        ami_scores = np.zeros((nr, nf), np.float32)

        for i, r in enumerate(self.rs):
            for j, freq in enumerate(self.freqs):
                print(r, freq)
                feats = self.compute_features(r, freq)
                D = self.compute_distance_matrix(feats)
                predict = self.kmeans(D)
                rand_scores[i, j] = adjusted_rand_score(ground_truth_labels, predict)
                ami_scores[i, j] = adjusted_mutual_info_score(ground_truth_labels, predict)

                if cluster_plots:
                    tsne = TSNE(n_components=3).fit_transform(D)
                    tsne_ax[i, j].scatter(tsne[:, 0], tsne[:, 1], c=predict)
                    umapper = umap.UMAP().fit(D)
                    umap.plot.points(umapper, labels=predict, ax=umap_ax[i, j])

                    tsne_ax[i, j].set_title(f"r={r}, fmax={freq}")
                    umap_ax[i, j].set_title(f"r={r}, fmax={freq}")

        if cluster_plots:
            tsne_fig.tight_layout()
            umap_fig.tight_layout()

        im1 = met_ax[0, 0].matshow(rand_scores, vmin=0., vmax=1.)
        met_ax[0, 0].set_title("Adj. Rand Score")
        plt.colorbar(im1, ax=met_ax[0, 0])


        im2 = met_ax[1, 0].matshow(ami_scores, vmin=0., vmax=1.)
        met_ax[1, 0].set_title("Adj. MI Score")
        plt.colorbar(im2, ax=met_ax[1, 0])

        met_fig.tight_layout()

    def grid_plot(self, r, freq):
        feats = self.compute_features(r, freq)
        D = self.compute_distance_matrix(feats)
        predict = self.kmeans(D)

        tsne_fig, tsne_ax = plt.subplot()
        umap_fig, umap_ax = plt.subplot()

        tsne = TSNE(n_components=3).fit_transform(D)
        tsne_ax.scatter(tsne[:, 0], tsne[:, 1], c=predict)
        umapper = umap.UMAP().fit(D)
        umap.plot.points(umapper, labels=predict, ax=umap_ax)

        tsne_ax.set_title(f"r={r}, fmax={freq}")
        umap_ax.set_title(f"r={r}, fmax={freq}")

        tsne_fig.tight_layout()
        umap_fig.tight_layout()


    def cluster_metrics(self):
        assert len(self.freqs) == 1
        nr = len(self.rs)

        ground_truth_labels = self.ground_truth_labels()

        rand_scores = np.zeros(nr, np.float32)
        ami_scores = np.zeros(nr, np.float32)

        fig, ax = plt.subplots(2)

        for i, r in enumerate(self.rs):
            feats = self.compute_features(r, self.freqs[0])
            D = self.compute_distance_matrix(feats)
            predict = self.kmeans(D)
            rand_scores[i] = adjusted_rand_score(ground_truth_labels, predict)
            ami_scores[i] = adjusted_mutual_info_score(ground_truth_labels, predict)

        ax[0].plot(self.rs, rand_scores)
        ax[0].set_xlabel("SVs kept")
        ax[0].set_ylabel("Adjusted Rand Score")

        ax[1].plot(self.rs, ami_scores)
        ax[1].set_ylabel("Adjusted MI Score")

        fig.suptitle(f"Frequency range: [0, {self.freqs[0]}] Hz\n{self.nclusters} units")

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


f = Fingerprinter(wfs, reconstruct_from_svd_clip,  np.arange(20), np.arange(2, 12), [8000])

f.cluster_metrics()

# generate the stack of hankel traj matrices
def hank_stack(w):
    ntr, ns = w.shape
    L = int(ntr // 2)
    K = int(ntr - L)

    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(ns, d=1/30_000)

    hanks = np.stack([hank_mat(W, f, L, K) for f in range(0, len(freqs))])
    return hanks





        
    



    
