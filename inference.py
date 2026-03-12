from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np, torch, neurokit2 as nk

from models import *
from loaders import *
from utils import *
import os 

#device = torch.device("cuda" if torch.cuda.is_available else "cpu") 
device = torch.device("cpu")
dataset = get_dataset(test_name, "test", sample_rate)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

models = {
    "stft": STFTUNet(),
    "unet": UNet(in_channels=2, out_channels=2, init_features=32),
    "conv-tasnet": ConvTasNet(),
    "sepformer": Sepformer(),
}
#ckpt_name = f"ckpt/{data_name}_{model_name}.pt"

nn_name = "2025-01-22_10h22min.pt"
ckpt_name = os.path.join("C:\\","Users",pc_name,"OneDrive - TMNA",
                        "ML1","wav2ecg_cavity_orig","ckpt",nn_name)

model = models[model_name].to(device)
#model.load_state_dict(torch.load(ckpt_name))
model.load_state_dict(torch.load(ckpt_name, map_location=torch.device('cpu')))

# inference on test data
model.eval()
ecg_preds = []
ecgs = [] ##array to store the ground truth 
names_list = []

for pcg, ecg, pcg_name in tqdm(loader):
    if "tasnet" in model_name:
        ecg_pred = model(pcg.to(device))
    else:
        ecg_pred = model(pcg.to(device))
    if model_name == "sepformer" or model_name == "speech-convtasnet":
        ecg_pred = ecg_pred[:, 0]
    else:
        ecg_pred = ecg_pred[0]
    ecg_pred = ecg_pred.cpu().detach().squeeze()
    ecg_pred = nk.ecg_clean(ecg_pred, sampling_rate=sample_rate)
    ecg_preds.append(ecg_pred)

    ecg = ecg_pred.squeeze()
    ecg = nk.ecg_clean(ecg, sampling_rate=sample_rate)
    ecgs.append(ecg)

    #retrieving name associated with inference file 
    #pcg_name contains address of one file at a time
    #segments is a list where each element is a part of the total directory between '\\'
    segments = pcg_name[0][0].split('\\')
    #s is the last part of that list, which is the name of the file minus extension
    s = segments[-1].replace(".wav", "")

    names_list.append(s)

# save the predictions
dir_save = os.path.join("C:\\","Users",pc_name,"OneDrive - TMNA",
                        "ML1","wav2ecg_cavity_orig","results")

preds_name = f"v2_{nn_name[:-3]}_{data_name}_on_{test_name}_{model_name}_pred.npy"
ecgs_name = f"v2_{nn_name[:-3]}_{data_name}_on_{test_name}_{model_name}_true.npy"

#np.save(preds_name, np.stack(ecg_preds))
np.save(os.path.join(dir_save, preds_name), np.stack(ecg_preds))
np.save(os.path.join(dir_save, ecgs_name), np.stack(ecgs))

print("Inference finished. Predictions saved.")
