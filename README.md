# ssa_denoise

Singular spectrum analysis is a denoising and signal reconstruction technique that encodes temporal and spatial signal lags. [Sternfels et al 2015](https://www.cgg.com/sites/default/files/2020-11/cggv_0000022842.pdf). This is particularly relevant to raw ephys waveforms, which can appear across multiple channels with a lag. SVD is performed on a matrix of shifted copies of the signal and the denoised/reconstructed waveform is recovered by selecting the top few singular values and frequencies. [Baluja and Covell 2006](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/32685.pdf) from Google Research describe a method of create sparse binary “fingerprints” from snippets of audio in frequency space and then using locality-sensitive hashing (LSH) to match waveforms to a list of known audio samples, in a process similar to Shazam’s song identification algorithm. An idea could be to see whether the Fx-space SVD components and singular values might be analogous to these spectrogram snippets and amenable to the same type of finger printing, which could be used as a toy neuron clustering model. I propose to a) evaluate the denoising/compression possibilities of SSA on ephys data, OR b) see if there is some way to imitate the “fingerprinting” process on compressed representations of waveforms. 

Examples wfs: https://drive.google.com/file/d/17Ha9FIp-gk4R0iWA4F95S6Qfc7hB_pgG/view?usp=drive_link

The file above has preprocessed waveforms for 13 datasets (10 units per dataset, 500 waveforms per unit).

So e.g. `arr[:500, :, :]` will give you the waveforms from the 1st unit of the 1st dataset. `arr[9*500:10*500,:,:]` the 10th unit of the 1st waveform etc

`Shape = (500*13*10, 121, 40)`
