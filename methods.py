'''
Auxiliary methods to declutter the main scripts 
'''
import matplotlib.pyplot as plt 
from utils import *              # global parameters
from auraloss import time
import os 
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import pickle
from scipy.signal import find_peaks
import torchaudio
from copy import deepcopy
import random 

#hello world method to check if this script is being imported 
def test_methods():
    print('methods.py activated')


def print_nn_info(device, model_name, total_params):
    print("=========================================================")
    print("Initializing Multi-channel NN:")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Num params: {total_params}")
    print("=========================================================")

'''Plotting data '''

#sets up training and valid plots that will be updated in real-time
def setup_training_plots():
    train_losses = []
    val_losses = []

    plt.ion()
    fig, axes = plt.subplots(2,1, figsize=(9,6))
    plt.subplots_adjust(hspace=0.6)
    font_axes = 14

    axes[0].clear()
    axes[0].set_title('Train Loss', fontsize=font_axes)
    axes[0].set_xlabel('Epochs', fontsize=font_axes)
    axes[0].xaxis.set_label_coords(.98, -0.2)
    axes[0].tick_params(labelsize=font_axes)

    axes[1].clear()
    axes[1].set_title('Validation Loss', fontsize=font_axes)
    axes[1].set_xlabel('Epochs', fontsize=font_axes)
    axes[1].xaxis.set_label_coords(.98, -0.2)
    axes[1].tick_params(labelsize=font_axes)

    plt.pause(0.1)

    return train_losses, val_losses, axes


# real-time updates on training and validation losses 
def update_training_plots(avg_loss_train, avg_loss_valid, train_losses, val_losses, axes):
    #plotting train and validation losses
    train_losses.append(avg_loss_train)
    val_losses.append(avg_loss_valid)

    axes[0].clear()
    axes[0].plot(train_losses)
    axes[0].set_title('Train Loss')

    axes[1].clear()
    axes[1].plot(val_losses)
    axes[1].set_title('Validation Loss')
    #pause was increased becasue plot was not updating correctly 
    plt.pause(0.1)



def plot_inference(ecg_pred, ):
    plt.plot(ecg_pred ,label='ecg_pred', color='r')
    # Customize the plot (add labels, title, legend, etc.)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Multiple Waveforms')
    plt.legend()
    # Show the plot
    plt.show()


'''Calcualte NN loss '''

def calculate_loss(multichannel_pred, ecg):
    #in ca-dense-unet, output is 17-CH files
    #System loops over all multi-channels to calculate loss and pick the lowest one
    if model_name == "ca-dense-unet":
        loss_list = []
        for index in range(ca_channels):

            # multichannel_pred is a tuple 
            # multichannel_pred[0] = signal, 
            # multichannel_pred[1] = noise
            # noise and signal are tensors with as many rows as there are channels 
            ecg_pred = multichannel_pred[0][:, index, :]

            # each loss_list item is calculated using all batch 
            # elements and one of the 17 channels
            loss_list.append(time.LogCoshLoss()(ecg_pred, ecg))
        #fidning the lowest loss, to use for backprop
        # min_value, min_index = min((tensor.item(), idx) for idx, tensor in enumerate(loss_list))        
        # loss = loss_list[min_index]

        #using approach 1.1: loss comes from channel 1Y, always
        loss = loss_list[1]
        min_value = 1
        min_index = 1
    
    #in ca-dense-unet-cnn, loss is expected to be just one single output, 
    #with same size as ground truth 
    elif model_name == "ca-dense-unet-cnn":
        #converting output to size [batch_Size, 8000], to match gorund truth
        dim1 = multichannel_pred.shape[0]
        dim2 = multichannel_pred.shape[2]
        ecg_pred = torch.reshape(multichannel_pred, (dim1,dim2))
        loss = time.LogCoshLoss()(ecg_pred, ecg)
        min_index = 0
    
    #this applies for "ca-dense-unet-cnn-cbp"
    #output = [batch 8000]
    else:
        dim1 = multichannel_pred.shape[0]
        dim2 = multichannel_pred.shape[1]
        ecg_pred = torch.reshape(multichannel_pred, (dim1,dim2))
        loss = time.LogCoshLoss()(ecg_pred, ecg)
        min_index = 0

    #this is to later save this info on a text dcument
    loss_name = 'time.LogCoshLoss()(ecg_pred, ecg)' 
    return loss, loss_name, min_index

def get_double_loss(ecg, multichannel_pred, pcg_name, device):
    '''
    Two loss functions will be calculated here:
    - loss_1: The first one will compare the waveform estimation from NN to the 
    ground truth (raw ecg, or wide gaussian etc.).
    - loss_2: The second one will compare the peak detection from the NN to the 
    ground truth. These peaks are generated using the get_peaks method
    '''

    #converting output to size [batch_Size, 8000], to match gorund truth
    #dim1 = multichannel_pred.shape[0]
    #dim2 = multichannel_pred.shape[2]
    ecg_pred = multichannel_pred #torch.reshape(multichannel_pred, (dim1,dim2))

    #denoising before pointwise 
    loss_1 = time.LogCoshLoss()(ecg_pred, ecg)

    #denoising signal before peak detection
    multichannel_pred = lowpass_filter(multichannel_pred)

    #list of ndarray containing nn peaks 
    nn_peaks_np = []
    #list of ndarray containing real ecg peaks
    hr_peaks_np = []
    # Iterate over array elements and tuple indices
    for i in range(len(pcg_name[0])):
        peaks, pos = get_peaks(multichannel_pred[0], pcg_name[0][i],device,ecg)
        nn_peaks_np.append(peaks)
        hr_peaks_np.append(pos)

    #stacking all elements in a single array, becasue it's faster to 
    #later convert to tensor
    stacked_nn_peaks = np.stack(nn_peaks_np)
    stacked_hr_peaks = np.stack(hr_peaks_np)

    #converting to tensor to send to gpu if needed
    nn_peaks = torch.from_numpy(stacked_nn_peaks).to(device)/sample_rate
    hr_peaks = torch.from_numpy(stacked_hr_peaks).to(device)/sample_rate

    sq_diff = (nn_peaks - hr_peaks) ** 2
    #obtaining the average of the sum of all errors 
    #loss_2 = sq_diff.sum(dim=1, keepdim=True).mean()
    #loss_2.requires_grad  = True

    loss = loss_1 #loss_2 * loss_1

    loss_name = 'time.LogCoshLoss()' #* sum of squared errors'  #'sum of squared errors'

    return loss, loss_name, nn_peaks, hr_peaks

def get_peaks(estimation, str_pcg_name,device,ecg):
    '''
    This function gets the peaks from the NN estimation
    - max hr in humans: 220 bpm minus age
    - for an 18 years old: 202bpm = 13.5 beats per 4 secs = 1beat per 296ms
    - at 2 kHz: 1 px = 0.5ms, so min distance = 296ms/.5ms = 592 px

    Args: 
        estimation: 1D nparray of NN estimation
        str_pcg_name: name of pcg file, 
            will be used to load csv containing position of heartbeats
        

    '''

    #normalizing the NN estimation
    estimation = estimation.squeeze().cpu().detach().numpy()
    #estimation = estimation.squeeze().detach().numpy()
    estimation = estimation / np.max(np.abs(estimation))

    #finding peaks in the NN estimation
    nn_peaks, _ = find_peaks(estimation, height=peak_threshold, distance = 592)
    hr_pos, _ = find_peaks(ecg[0], height=peak_threshold, distance = 592)
    #loading the real peaks
    #detatching file name from the directory
    #segments = str_pcg_name.split('\\')
    #adjusting first segment from c: to c:\\, to work on os.path.join
    #segments[0] = segments[0] + '\\'
    #formulating the file name that contains positions of HR peaks
    #str_filename = '[HR_POS]'+segments[-1][:-3] + 'csv'

    #unpack all elements from segments except last one
    #dir = os.path.join(*segments[:-1])
    #dir = os.path.join(dir, str_filename)

    #loading heart rate peaks
    #hr_pos = np.genfromtxt(dir, delimiter="")
    #hr_pos = hr_pos * sample_rate 

    #adjusting vector size to contain fixed amount of hr coordinates:
    nn_peaks = adjust_array(nn_peaks)
    hr_pos = adjust_array(hr_pos)



    return nn_peaks, hr_pos


'''LOADING DATA'''

def load_wav_to_tensor(dir_noise):
    '''
    Scans a folder for all the wav files on it 
    Each wav file contains multi-channel data
    All data will be concatenated to one tensor of shape 
    [num_channels, total_length]
    
    Parameters: 
        - dir_noise: directory path where wav files are stored 
    '''
    noise = torch.empty(ca_channels,0)
    #initializing list that will contain path to each file (dir/filename.csv)
    noise_paths = []

    #listing all files in the directory
    for filename in os.listdir(dir_noise):
        if filename.endswith(".wav"):
            if filename.startswith("[NOISE]"):
                #extracting path of a single noise file 
                noise_path = os.path.join(dir_noise, filename)
                
                #loading a single noise file as tensor
                noise_file, _ = torchaudio.load(noise_path)

                #concatenating all noise together
                noise = torch.cat((noise, noise_file), dim=1)

                #creating (dir/filename.wav) for each wav file in dir_noise
                #and appending all paths to list 
                noise_paths.append(noise_path)

    return noise

def extract_noise_snippet(all_noise):
    '''
    Extracts a multi-channel 4 second snippet from the noise source, at random.
    - all_noise: 17 rows of noise source
    '''
    upper_lim = all_noise.shape[1] - ca_time - 1

    start_index = torch.randint(0, upper_lim, (1,))

    noise = all_noise[:, start_index:(start_index+ca_time)]

    return noise


def mixing_signal_and_noise(signal_batch, noise_batch, snr_db):
    '''
    This function will mix signal and noise at a certain SNR 
    1. Noise is a random snippet from all_noise
    2. original SNR between noise and signal is calculated for each row in signal data
    3. based on orig snr, the linear ratio between signal and noise is calculated 
    4. using new snr defined in snr_db, the new desired lin ratio is calculated 
    5. adjusted noise is defined as: new_noise = sqrt(old_noise^2 * linratio_old/linratio_new) 
    '''

    #list that will contain the noisy input
    #each list element is a tensor of shape [17, 8k]
    #list length is the batch_size
    input_list = []

    #looping over the all [17, 8000] signals in the data batch 
    for index in range(signal_batch.shape[0]):

        #extracting multi-channel input and noise 
        signal = signal_batch[index,:,:]
        noise  = noise_batch[index,:,:]

        #power of original noise waveform 
        npow_old = torch.mean(torch.pow(noise, 2))
        #power of signal waveform
        spow_old = torch.mean(torch.pow(signal,2))
        #original snr based on signal and noise 
        snr_old = 10*torch.log10(spow_old / npow_old)
        #linear ratio between signal and noise based on original waveforms
        linratio_old = 10 ** (snr_old/10)

        #desired linear ratio is the value defined by snr_db variable 
        linratio_new = 10 ** (snr_db/10)

        # #calculating the required noise power to achieve such snr 
        # npow_new = npow_old * (linratio_old / linratio_new)

        #calculating the new noise value to achieve the desired SNR 
        noise_new = noise * torch.sqrt(linratio_old / linratio_new)

        #choosing a random number between dropout_ch and ca_channels
        #this value will define how many noisy channels are 0 
        num_clear_ch = random.randint(dropout_ch, ca_channels)

        #implementing channel dropout in the noisy input 
        #indices = torch.randperm(dropout_ch)[:ca_time]
        indices = torch.randperm(num_clear_ch)[:ca_time]

        #at random, a dorpot_ch number of channels will NOT have noise
        #added to it 
        noise_new[indices,:] = 0

        #mixing signal and noise at correct ratio, then normalizing
        noisy_input = signal + noise_new
        noisy_input = noisy_input/ torch.abs(torch.max(noisy_input))

        #appending the tensor to a list 
        input_list.append(noisy_input)

        # #proofing that the SNR works, testing for SNR_DB = 3dB
        # #SNR = 10log(mean(S1**2) / mean(S2**2))
        # testing_snr = 10*torch.log10(spow_old/torch.mean(torch.pow(noise_new, 2)))
        # print(f"calculated SNR with the adjusted noise: {testing_snr}dB")
    
    #stacking all tensors into a batch of noisy inputs
    noisy_batch = torch.stack(input_list, dim=0)

    return noisy_batch

def get_noisy_input(all_noise, pcg, snr_lvl):
    #creating a list, where each element is a [17, 8000] noise tensor
    #lenght of list is same as batch_size
    noise_list = [extract_noise_snippet(all_noise) for _ in range(batch_size)]

    #stacking all tensors into a batch of noise tensors 
    noise_batch =  torch.stack(noise_list, dim=0)
    
    #mixing batch of inputs with batch of noise, 
    #at each time a new batch is loaded
    noisy_pcg = mixing_signal_and_noise(pcg, noise_batch, snr_db=snr_lvl)

    return noisy_pcg

def lowpass_filter(orig_tensor):
    '''
    denoising the NN estimation before calculating loss
    this is based on the fact that ground truth (wide gaussian)
    has no components over 20Hz. This is meant as a replacement for
    nk.ecg_clean() that used to be implemented before 
    '''
    cutoff_freq = 20
    clean_tensor = torchaudio.functional.lowpass_biquad(orig_tensor, sample_rate, cutoff_freq)
    return clean_tensor
    
    

def adjust_array(array):
    '''
    This function is used to convert the arrays cotaning heartbeats 
    into a 14 channel size, by filling it with zeros or clipping their size 
    when they go over 14 elements'''
    if array.shape[0] < 14:
        zeroes_array = np.zeros(14-array.shape[0])
        adjusted_array = np.concatenate((array, zeroes_array))
    else:
        adjusted_array = array[:14]
    return adjusted_array

def get_mae(nn_peaks, hr_peaks):
    '''
    Calculates MAE using peaks of NN estiamtion and HR peaks 
    nn_peaks: array containing position of the hr peaks 


    '''
    #converting to numpy 
    hr_array = hr_peaks.to(torch.device("cpu")).squeeze().detach().numpy()
    nn_array = nn_peaks.to(torch.device("cpu")).squeeze().detach().numpy()

    #removing zeros at the end of the array only: 
    hr_array = np.trim_zeros(hr_array, 'b')
    nn_array = np.trim_zeros(nn_array, 'b')

    if hr_array.shape[0] != nn_array.shape[0]:
        hr_match = False
        mae = 0
    else:
        hr_match = True 
        nn_hri = nn_array[1:] - nn_array[:-1]
        hr_hri = hr_array[1:] - hr_array[:-1]
        mae  = np.mean(np.abs(nn_hri - hr_hri))

    return mae, hr_array, nn_array, hr_match

def get_global_mae(hr_match_list, nn_peaks_list, hr_peaks_list, dir_mae):
    
    '''
    This method calculates global MAE for all the instances where 
    HR calculation was correct. This info is saved on file "global_analysis",
    together with the ratio of files where HR detection was right.

    Args:
        hr_match_list: list of booleans describing all cases where HR calculation is right
        nn_peaks_list: list of NN estimations. Each element contains the location of 
            hr peaks for one input 
        hr_peaks_list: list of real HR locations. Each element contains locatino of 
            real hr peaks for one input 
        dir_mae: location to save files 
    

    '''
    #store all the instances where hr estimation was correct
    right_hr = 0
    #get all the hr estimations
    total_hr = len(hr_match_list)

    right_nn_peaks = np.empty(0)
    right_hr_peaks = np.empty(0)

    #looping over hr_match_list to identify when the number of HR detected
    #match the correct number of HR 
    for p in range(len(hr_match_list)):

        #if number of hr is a match, the position of the HR in NN estimation
        #and in real HR will be contatenated to bigger np arrays
        if hr_match_list[p] == True:

            #updating the number of cases where HR detection is right 
            right_hr = right_hr+1

            #updating np arrays with position of the correct HR estimations
            right_nn_peaks = np.concatenate((right_nn_peaks, nn_peaks_list[p]))
            right_hr_peaks = np.concatenate((right_hr_peaks, hr_peaks_list[p]))

    #calculating global MAE from all the correct beats
    global_mae = np.mean(np.abs(right_nn_peaks - right_hr_peaks))

    correct_hr_ratio = right_hr / total_hr

    text_content = f"""
    Group MAE: {global_mae}
    Group Ratio of correct HR detection: {correct_hr_ratio}
    """

    text_name = os.path.join(dir_mae, "local_analysis.txt")

    #saving text
    with open(text_name, 'w') as file: 
        file.write(text_content)
    return global_mae, correct_hr_ratio

'''
Methods to save NN parameters
'''

# saving info about NN parameters in a text document
def save_nn_description(total_params, loss_name, optimizer,dir_root, nn_name, observations):
    #saving description of parameters adopted:
    text_content = f"""
    Architecture: {model_name}
    NN size: {total_params} parameters
    dataset: {data_name}
    batch size: {batch_size}
    learning rate: {learning_rate}
    number of epochs: {num_epochs}
    loss function: {loss_name}
    optimizer: {optimizer.__class__.__name__}
    sample rate: {sample_rate}
    Multichannel parameters: 
    window length: {window_length}
    center: {center}
    K = {K}
    d = {d}
    Unet layers: {U_net_layers}
    -----------------------------------------
    Spectrogram specs:
    n_fft: {n_fft}
    hop_length: {hop_length}
    Window length: {window_length}
    Window: {window}
    ----------------------------------------
    Dataset:
    - Location: {dataset_folder}
    - Batch size: {batch_size}
    - input is using 17 channels
    - Ground truth: wide gaussian
    - Denoising with  bandpass [8 128] + MIRISE WT algorithm
    - Uses 3 datasets: 01_17 + 07_11 + 10_02
    ----------------------------------------
    Observations: 
    {observations}
    """ 
    
    #creating text directory
    dir_text = os.path.join(dir_root, "hyperparam_info")
    create_dirs(dir_text, "save all info")

    text_name = os.path.join(dir_text, nn_name + ".txt")

    #writing text
    with open(text_name, 'w') as file: 
        file.write(text_content)

def create_dirs(dir_name, purpose):
    #checking if a dir exists. if not, create it 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        #print("Directory to", purpose, "created successfully")
        #print(dir_name)
    else:
        print("Directory to", purpose, "already exists")


def name_nn(now, model):
    str_date = now.strftime("%Y-%m-%d][%Hh%Mmin")

    nn_name = f'[{str_date}][{cavity_data}]'
    #C:\Users\Admin\Desktop\hr_hri\wav2ecg\inference_tests
    dir_root = os.path.join("C:\\", "Users","Admin", "Desktop", "hr_hri",
                            "wav2ecg", "inference_tests", nn_name)
    
    #creating main dir to save all things 
    create_dirs(dir_root, "save all NN-related files")

    return dir_root, nn_name

def name_nn_with_flag(nn_creation_date, nn_flag):
    str_date = nn_creation_date.strftime("%Y-%m-%d][%Hh%Mmin")

    suffix = ""

    if "[wide_gaussian]" in dataset_folder:
        suffix = "[wide_gaussian]"
    elif "[narrow_gaussian]" in dataset_folder:
        suffix = "[narrow_gaussian]"
    elif "whole_ecg" in dataset_folder:
        suffix = "[whole_ecg]"
    elif "r_wave_only" in dataset_folder:
        suffix = "[r_wave_only]"
    
    if "[K1]" in dataset_folder:
        suffix = suffix + "[K1]"
    elif "[K2]" in dataset_folder:
        suffix = suffix + "[K2]"
    elif "[K3]" in dataset_folder:
        suffix = suffix + "[K3]"
    elif "[K4]" in dataset_folder:
        suffix = suffix + "[K4]"
    elif "[K5]" in dataset_folder:
        suffix = suffix + "[K5]"
    

    nn_name = f'[{str_date}]{suffix}[{nn_flag}]'
    #C:\Users\Admin\Desktop\hr_hri\wav2ecg\inference_tests
    dir_root = os.path.join("C:\\", "Users","Admin", "Desktop", "hr_hri",
                            "wav2ecg", "inference_tests", nn_name)
    
    #creating main dir to save all things 
    create_dirs(dir_root, "save all NN-related files")

    return dir_root, nn_name

def save_training_plot(dir_root, nn_name):
    
    #creating plot directory 
    dir_plots = os.path.join(dir_root, "training_plots")
    create_dirs(dir_plots, "save all training plots")

    #saving training plot in two formats
    plt.savefig(os.path.join(dir_plots, nn_name + ".JPG"), format='jpg')
    plt.savefig(os.path.join(dir_plots, nn_name + ".svg"), format='svg')

def save_ckpt(nn_state_dict, dir_root, nn_name):
    dir_ckpt = os.path.join(dir_root, "ckpt")
    create_dirs(dir_ckpt, "save all checkpoints")

    ckpt_name = os.path.join(dir_ckpt, nn_name + ".pth")

    torch.save(nn_state_dict, ckpt_name)


def save_inference(dir_root, nn_name, ecg_preds, ecg_real, names_list):
    # save the predictions
    dir_npy = os.path.join(dir_root, "npy")
    create_dirs(dir_npy, "saving numpy files")    

    preds_name = os.path.join(dir_npy,'[PRED_ECG]'+nn_name)


    real_name = os.path.join(dir_npy, '[REAL_ECG]'+nn_name)

    #saving name of files used for inference
    inf_name = os.path.join(dir_npy, '[NAMES_INF]'+nn_name+".pkl")

    #f"results/{data_name}_on_{test_name}_{model_name}_test.npy"
    np.save(preds_name, np.stack(ecg_preds))
    np.save(real_name, np.stack(ecg_real))

    # Save the list of file names
    with open(inf_name, "wb") as file:
        pickle.dump(names_list, file)
    
    print(f"Predictions saved at: {preds_name}")


def get_latest_subfolder(directory):
  """
  This function returns the name of the most recent sub-folder within a directory.
  Args:directory (str): Path to the directory to scan.
  Returns:str: Name of the most recent sub-folder, or None if no sub-folders are found.
  """
  # Get all sub-directories within the specified directory
  subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

  # Check if any sub-folders were found
  if not subfolders:
    return None

  # Use max() with key argument to find the sub-folder with the latest modification time
  latest_folder = max(subfolders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

  return latest_folder


''' 
Testing algorithms to measure performance
with different datasets
'''


def shrink_perturb(models, current_model, device, shrink, perturb):
    '''
    Implementation of the algorithm shown in : On Warm-Starting Neural Network Training
    Whenever a new dataset is added, the following happens: 
    - All weights are shrank in half
    - A random value from normal distribution is added to all weights 
    Code based on: https://github.com/JordanAsh/warm_start/blob/main/run.py#L16
    '''
    #creating a new model with same size and num weights as the current model
    new_model = models[model_name]
    new_init = torch.nn.DataParallel(new_model, device_ids=[0,1,2])
    new_init.to(device)

    params1 = new_init.parameters()
    params2 = current_model.parameters()

    #looping over elements of new_init and current weights at the same time
    for p1, p2 in zip(*[params1, params2]):
        #updating values from new_init
        #deepcopy creates a completely independent copy 
        p1.data = deepcopy(shrink * p2.data + perturb * p1.data)

    # #evaluating new parameters 
    # params3 = new_init.parameters()
    # flag = 0
    # #looping over elements of new_init and current weights at the same time
    # for p3, p2 in zip(*[params3, params2]):
    #     if torch.equal(p3.data, p2.data):
    #         print("same value")
    #         flag += 1
    # print(flag)
    return new_init


