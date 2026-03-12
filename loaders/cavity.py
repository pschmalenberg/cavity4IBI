from torch.utils.data import Dataset
import numpy as np, random, os
import torchaudio, pandas as pd
from torchaudio import transforms
from scipy.io import loadmat
import torch, neurokit2 as nk
import os 
import utils as u


class CavityDataset(Dataset):
    def __init__(self, path, split, sr, use_ecg=True):
        self.path = path
        
        self.path_input = os.path.join(self.path, "[input]")
        self.path_label = os.path.join(self.path, "[wide_gaussian]")
        
        self.split = split
        self.sr = sr
        self.use_ecg = use_ecg

        self.resample = transforms.Resample(4000, self.sr)

        self.subpaths_label = [os.path.join(self.path_label, i) for i in os.listdir(self.path_label)]
        self.subpaths_input = [os.path.join(self.path_input, i) for i in os.listdir(self.path_input)]

        self.subpaths_label = self.isolate_split(self.subpaths_label)
        self.subpaths_input = self.isolate_split(self.subpaths_input)
        
        #self.subpaths = self.isolate_split(self.subpaths)
        self.pcg_paths = [f for f in self.subpaths_input if f.endswith(".wav")]
        self.ecg_paths = [f for f in self.subpaths_label if f.endswith(".csv")]

        self.pcg_paths.sort()
        self.ecg_paths.sort()


    def isolate_split(self, files):
        if self.split == "train":
            return [f for f in files if "[A]" in f]
        elif self.split == "valid":
            return [f for f in files if "[B1]" in f]
        elif self.split == "test":
            return [f for f in files if "[B2]" in f]

    def process_pcg(self, pcg, name):
        pcg = torch.FloatTensor(pcg)
        
        # self.segment defines the overlap between 
        # consecutive samples
        pcg = self.segment(self.resample(pcg))
        for n, seg in enumerate(pcg):
            # PCG filtering
            seg = nk.signal_filter(seg, lowcut=25, highcut=50, sampling_rate=self.sr)
            seg = torch.FloatTensor(seg.copy()).unsqueeze(0)

            #index is always 3 digits: 001, 042, 114, ...
            #facilitates alphabetical ordering 
            index = str(n).zfill(3)

            # save new file
            outpath = os.path.join(self.clean, name.replace(".wav", f"_{index}.wav"))
            torchaudio.save(outpath, seg, sample_rate=self.sr)

    def process_ecg(self, ecg, name):
        ecg = torch.FloatTensor(ecg)
        ecg = self.segment(self.resample(ecg))
        for n, seg in enumerate(ecg):
            # ECG cleaning and filtering
            seg = nk.ecg_clean(seg, sampling_rate=self.sr)

            #index is always 3 digits: 001, 042, 114, ...
            #facilitates alphabetical ordering 
            index = str(n).zfill(3)

            # save new file
            ecg_path = os.path.join(self.clean, name.replace(".wav", f"_{index}.csv"))
            pd.DataFrame(seg).to_csv(ecg_path)

    def segment(self, signal, dur=4):
        n_samples = int(dur * self.sr)
        
        # using NO overlap if the data is the one to be plotted
        if self.split == "test":
            n_hops = int(n_samples)
        else:
        # using overlap = 50%
            n_hops = int(n_samples / 2)

        n_segments = int(len(signal) / n_hops)
        segments = [
            signal[i * n_hops : i * n_hops + n_samples] for i in range(n_segments - 1)
        ]
        # discard last segment if not complete
        return [s for s in segments if len(s) == n_samples]

    def __len__(self):
        return len(self.pcg_paths)

    def __getitem__(self, idx):
        pcg, _ = torchaudio.load(self.pcg_paths[idx])
        pcg = pcg.squeeze() / pcg.max()

        pcg_name = []
        pcg_name.append(self.pcg_paths[idx])

        # add noise to the PCG
        if self.split == "train":
            snr_list = np.linspace(0, 30, 100)
            snr_db = random.choice(snr_list)
            pcg = self.add_noise(pcg.numpy(), snr_db)

        ecg = pd.read_csv(self.ecg_paths[idx])
        ecg = torch.FloatTensor(ecg.iloc[:,1].values)

        #truncating elements in case there are more than 8001 points
        ecg = ecg[:8000]
        pcg = pcg[:8000]

        if not self.use_ecg:
            peaks = nk.ecg_peaks(ecg.numpy(), sampling_rate=self.sr)
            peaks = torch.FloatTensor(peaks[0])
            return pcg, peaks
        else:
            return pcg, ecg, pcg_name

    def add_noise(self, pcg, snr_db=30):
        # Generate noise with the same length as PCG
        noise = np.random.normal(0, 1, len(pcg))
        # Calculate the power of the PCG
        power = np.square(pcg).mean()
        # Calculate the power of the noise in dB
        noise_power = power / (10 ** (snr_db / 10))
        # Normalize the noise to the desired power
        noise *= np.sqrt(noise_power / np.mean(np.square(noise)))
        # Add the noise to the PCG
        return torch.FloatTensor(pcg + noise)


if __name__ == "__main__":
    path = os.path.join("C:\\","Users",u.pc_name,"OneDrive - TMNA",
                            "ML1","wav2ecg_cavity_orig","cavity_data")
    dataset = CavityDataset(path, "train", sr=2000)
    print(len(dataset), dataset[0][0].shape, dataset[0][1].shape)
