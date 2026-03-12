from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from torchaudio import transforms
from auraloss import time, freq
from tqdm import tqdm
import math, torch

from models import *
from loaders import *
from utils import *
from datetime import datetime
import methods as m

test_dataset = get_dataset(test_name, "test", sample_rate)
train_dataset = get_dataset(data_name, "train", sample_rate)
valid_dataset = get_dataset(data_name, "valid", sample_rate)


loaders = {
    "train": DataLoader(train_dataset, batch_size, shuffle=True),
    "valid": DataLoader(valid_dataset, batch_size, shuffle=False),
    "test": DataLoader(test_dataset, batch_size=1, shuffle=False),
}

classes = {
    "mirise": 2, "breathing": 4, "cavity_data": 2, "steth_exp": 4, "steth_all": 9, "all": 13}
n_classes = classes[data_name]
print(f"batch size: {batch_size}")

device = torch.device("cuda" if torch.cuda.is_available else "cpu") 

models = {
    "stft": STFTUNet(),
    "unet": UNet(in_channels=2, out_channels=2, init_features=32),
    "conv-tasnet": ConvTasNet(),
    #"speech-convtasnet": Speech_ConvTasNet(checkpoint),
    "sepformer": Sepformer(),
}

model = models[model_name]
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)

#############################
# Waveform <--> Spectrogram #
#############################
stft = transforms.Spectrogram(n_fft, hop_length=hop_length, power=None)
istft = transforms.InverseSpectrogram(n_fft, hop_length=hop_length)


def time2freq(wav, sqrt=True):
    """
    wav.shape = (batch_size, 1, n_samples)
    out.shape = (batch_size, 2, n_fft, n_hops)
    """
    spec = stft(wav.cpu())
    abs = torch.sqrt(spec.abs()) if sqrt else spec.abs()
    return torch.cat((abs, spec.angle()), 1).to(device)


def freq2time(spec):
    """
    spec.shape = (batch_size, 2, n_fft, n_hops)
    out.shape = (batch_size, 1, n_samples)
    """
    abs, ang = spec[:, 0], spec[:, 1]
    y = istft(torch.polar(abs, ang).cpu())
    return y.unsqueeze(1).to(device)


def pad2power(signal):
    length = signal.shape[-1]
    next_power = 2 ** (math.ceil(math.log2(length)))
    pad_size = next_power - length
    return pad(signal, (0, pad_size), "constant", 0)


def compute_loss(ecg_est_freq, ecg_tgt_time, orig_shape):
    # extract time-domain estimate
    ecg_est_time = freq2time(ecg_est_freq)
    ecg_est_time = ecg_est_time[..., :orig_shape]

    # define time + frequency objectives
    time_loss = time.LogCoshLoss()
    stft_loss = freq.STFTLoss()

    # calculate losses
    ecg_tgt_time = ecg_tgt_time.unsqueeze(1)
    this_time_loss = time_loss(ecg_tgt_time, ecg_est_time)
    this_stft_loss = stft_loss(ecg_tgt_time, ecg_est_time)

    # return the weighted objective
    return this_time_loss + 0.001 * this_stft_loss

timestamp_epoch = datetime.now()

for epoch in range(num_epochs):
    total_loss = {
        "train": 0,
        "valid": 0,
    }

    # run training batches
    for pcg, ecg, _ in tqdm(loaders["train"]):
        # Move data to device
        pcg = pcg.to(device)
        ecg = ecg.to(device)

        # Forward Pass
        # ecg_est_freq = model(pcg_freq)
        ecg_pred = model(pcg)
        if model_name == "sepformer" or model_name == "speech-convtasnet":
            ecg_pred = ecg_pred[:, 0]
        else:
            ecg_pred = ecg_pred[0]

        # Backpropagation
        loss = time.LogCoshLoss()(ecg_pred, ecg)

        # this condition was created to check if calculations are returning 
        # NaN, which can happen if there's a corrupt file. 
        # The batch files involved will be printed and can be later deleted 
        if torch.isnan(loss):
            print("Loss is returning nan values. Here are the batch elements involved:")
            for element in pcg_name:
                print(element)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss["train"] += loss.item()

    # run validation batches
    for pcg, ecg, pcg_name in loaders["valid"]:
        # Move data to device
        pcg = pcg.to(device)
        ecg = ecg.to(device)

        # Forward Pass
        # ecg_est_freq = model(pcg_freq)
        ecg_pred = model(pcg)
        if model_name == "sepformer" or model_name == "speech-convtasnet":
            ecg_pred = ecg_pred[:, 0]
        else:
            ecg_pred = ecg_pred[0]

        loss = time.LogCoshLoss()(ecg_pred, ecg)

        # this condition was created to check if calculations are returning 
        # NaN, which can happen if there's a corrupt file. 
        # The batch files involved will be printed and can be later deleted 
        if torch.isnan(loss):
            print("Loss is returning nan values. Here are the batch elements involved:")
            for element in pcg_name:
                print(element)

        # Compute loss
        total_loss["valid"] += loss.item()

    # Print average loss for the epoch
    avg_loss_train = total_loss["train"] / len(loaders["train"])
    avg_loss_valid = total_loss["valid"] / len(loaders["valid"])

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_loss_train}")
    print(f"Valid Loss: {avg_loss_valid}\n")

    # Adjust lr based on validation loss
    scheduler.step(avg_loss_valid)

    # save the fine-tuned model
    now = datetime.now()
    delay_mins = (now - timestamp_epoch).total_seconds() / 60

    # saving nn checkpoint every two hours or when num_epochs ends 
    if delay_mins > 120 or epoch == (num_epochs-1):
        timestamp_epoch = now
        nn_name = now.strftime("%Y-%m-%d_%Hh%Mmin")

        #C:\Users\Admin\OneDrive - TMNA\ML1\wav2ecg_cavity_orig\ckpt 
        ckpt_name = os.path.join(dir_save,nn_name+".pt")

        torch.save(model.state_dict(), ckpt_name)
        print("Saving NN checkpoint")


print("Fine-tuning finished. Model saved.")
