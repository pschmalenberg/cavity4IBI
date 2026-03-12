from tqdm import tqdm
from scipy.io import loadmat
from torchaudio import transforms, load
import numpy as np, torch
import sys, neurokit2 as nk

from models import *
from loaders import *
from utils import *
import os

def segment(signal, sr, dur=4):
    n_samples = int(dur * sr)
    n_hops = int(n_samples / 10)
    n_segments = int(len(signal) / n_hops)
    segments = [
        signal[i * n_hops : i * n_hops + n_samples] for i in range(n_segments - 1)
    ]
    # pad segments if not complete
    for i in range(n_segments - 1):
        segments[i] = np.pad(segments[i], (0, n_samples - len(segments[i])))
    return segments


def process_pcg(pcg, init_sr):
    resample = transforms.Resample(init_sr, sample_rate)
    pcg = resample(torch.FloatTensor(pcg))
    orig_shape = pcg.shape[0]
    pcg = segment(pcg.numpy(), sr=sample_rate)

    pcg_list = []
    for seg in pcg:
        seg = nk.signal_filter(seg, lowcut=25, highcut=50, sampling_rate=sample_rate)
        seg = torch.FloatTensor(seg.copy()).unsqueeze(0)
        pcg_list.append(seg)
    return pcg_list, orig_shape


def process_ecg(ecg, init_sr):
    resample = transforms.Resample(init_sr, sample_rate)
    ecg = resample(torch.FloatTensor(ecg)).numpy()
    ecg = nk.ecg_clean(ecg, sampling_rate=sample_rate)
    return ecg


def merge_overlap(ecg_list, sr):
    window = len(ecg_list[0])
    hop = window // 10
    length = len(ecg_list) * hop + window

    ecg_final = np.zeros(length)
    for i, ecg in enumerate(ecg_list):
        ecg = nk.ecg_clean(ecg, sampling_rate=sr)
        ecg_final[i * hop : i * hop + window] += ecg

    # handle fading due to overlapping
    additional = np.ones(length)
    the_end = window - hop
    for i in range(length - the_end):
        additional[i * hop : i * hop + window] += 1
    additional[-the_end:] = additional[:the_end][::-1]

    return ecg_final.squeeze() / additional


device = torch.device("cpu")
models = {
    "stft": STFTUNet(),
    "unet": UNet(in_channels=2, out_channels=2, init_features=32),
    "conv-tasnet": ConvTasNet(),
    #"speech-convtasnet": Speech_ConvTasNet(checkpoint),
    "sepformer": Sepformer(),
}


str_filename = "cavity_data_conv-tasnet.pt"
ckpt_name = os.path.join("C:\\","Users","449443","OneDrive - TMNA",
                        "ML1","wav2ecg_cavity_orig","ckpt",str_filename)


#ckpt_name = f"ckpt/{data_name}_{model_name}.pt"

model = models[model_name].to(device)
model.load_state_dict(torch.load(ckpt_name, map_location=torch.device('cpu')))

# load data from command line
print("\nLoading", sys.argv[1])
test_name = sys.argv[1]
if test_name.endswith(".wav"):
    pcg, sr = load(test_name)
    pcg_list, orig_shape = process_pcg(pcg.squeeze(), sr)
elif test_name.endswith(".mat"):
    data = loadmat(test_name)["data"].T
    ecg_gt = process_ecg(data[0], 4000)
    pcg_list, orig_shape = process_pcg(data[1], 4000)

# inference on test data
print(f"Extracting ECG from {model_name}, trained on {data_name} data:")
model.eval()
ecg_preds = []
for pcg in tqdm(pcg_list):
    pcg = pcg.squeeze() / pcg.max()
    ecg_pred = model(pcg.to(device))[0]
    ecg_pred = ecg_pred.cpu().detach().squeeze()
    ecg_preds.append(ecg_pred.numpy())

# reconstruct the output ECG
print("Merging into final output...")
ecg_final = merge_overlap(ecg_preds, sr=sample_rate)[:orig_shape]
ecg_final = nk.ecg_clean(ecg_final, sampling_rate=sample_rate)

# save to ./results/ folder
test_name = test_name.split("/")[-1].split(".")[0]
np.save(f"results/{data_name}_on_{test_name}_{model_name}_pred.npy", ecg_final)
np.save(f"results/{data_name}_on_{test_name}_{model_name}_true.npy", ecg_gt)
print("Inference finished. Predictions saved.\n")
