#%%#####################################################################
# Imports
########################################################################
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np, torch, neurokit2 as nk
from models import *
from loaders import *
from utils import *
import methods as m     # auxiliary methods
from auraloss import time, freq
import pickle
import scipy.io as sio 
import csv 
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#%%#####################################################################
# Loading NN
########################################################################

#retrieving name and directory of last nn generated
# dir_path = os.path.join("C:\\", "Users","Admin", "Desktop", "hr_hri",
#                             "wav2ecg", "inference_tests")
dir_path = os.path.join(r"C:\Users\Admin\Desktop\2025 Cavity\2025 Neural Network\ckpt")

nn_name ='2025-11-22_23h02min' #raw ecg #'2025-10-17_16h30min' gaussian  #m.get_latest_subfolder(dir_path)
#loading a different nn version: 
#nn_name = '[2024-12-06][00h59min09sec]ca-dense-unet-cnn_200_epochs'

dir_root = os.path.join(dir_path, nn_name)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dataset = get_dataset(test_name, "test", sample_rate)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

models = {
    "stft": STFTUNet(),
    "unet": UNet(in_channels=2, out_channels=2, init_features=32),
    "conv-tasnet": ConvTasNet(),
    #"speech-convtasnet": Speech_ConvTasNet(checkpoint),
    "sepformer": Sepformer(),
}

print(f"inference_mc, nn_name: {nn_name}")
print(f"inference_mc, dir_root: {dir_root}")


#C:\Users\Admin\Desktop\hr_hri\wav2ecg\inference_tests\[2024-05-03][18h39min12sec]ca-dense-unet-cnn_2_epochs\cpkt
#'C:\\Users\\Admin\\Desktop\\hr_hri\\wav2ecg\\inference_tests\\[2024-05-03][18h39min12sec]ca-dense-unet-cnn_2_epochs\\ckpt\\[2024-05-03][18h39min12sec]ca-dense-unet-cnn_2_epochs.pth'
ckpt_name = os.path.join(dir_root, "ckpt", nn_name+".pth")

ckpt = torch.load(ckpt_name)

#loading a NN model
model = models[model_name]

#this is necessary if the model being loaded was also trained using 
#nn.DataParallel.
#model = torch.nn.DataParallel(model, device_ids=[0,1,2])

#loading weights saved as .pth
model.load_state_dict(ckpt)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())

#print NN info
m.print_nn_info(device, model_name, total_params)

#%%############################################
#inference on test data
###############################################
model.eval()
ecg_preds = []
ecg_real = []
names_list = []

hri_mae = []
nn_peaks_list = []
hr_peaks_list = []
hr_match_list = []
for pcg, ecg, pcg_name in tqdm(loader):
    pcg = pcg.to(device)
    
    multichannel_pred = model(pcg)
    multichannel_pred = multichannel_pred[0]

    #denoising estimation before calculating loss fct
    #multichannel_pred = m.lowpass_filter(multichannel_pred)

    ecg = ecg.to(device)

    _, _, nn_peaks, hr_peaks = m.get_double_loss(ecg, multichannel_pred, pcg_name, device)
    
    mae, hr_array, nn_array, hr_match = m.get_mae(nn_peaks, hr_peaks)

    hri_mae.append(mae)
    nn_peaks_list.append(nn_array)
    hr_peaks_list.append(hr_array)
    hr_match_list.append(hr_match)


    #retrieving name associated with inference file 
    #pcg_name contains address of one file at a time
    #segments is a list where each element is a part of the total directory between '\\'
    segments = pcg_name[0][0].split('\\')
    #s is the last part of that list, which is the name of the file minus extension
    s = segments[-1].replace(".wav", "")

    names_list.append(s)

    #converting output to size [batch_Size, 8000], to match gorund truth
    #dim1 = multichannel_pred.shape[0]
    #dim2 = multichannel_pred.shape[2]
    multichannel_denoised = m.lowpass_filter(multichannel_pred)
    ecg_pred = multichannel_denoised #torch.reshape(multichannel_denoised, (dim1,dim2))

    ecg_pred = ecg_pred.cpu().detach().squeeze()

    #filters noise from the prediciton ECG
    #ecg_pred = nk.ecg_clean(ecg_pred, sampling_rate=sample_rate)
    ecg_preds.append(ecg_pred)

    ecg_real.append(ecg[0].cpu().numpy())

#saving the np arrays
m.save_inference(dir_root, nn_name, ecg_preds, ecg_real, names_list)


str_preds = '[PRED_ECG]' + nn_name + '.npy'
str_real =  '[REAL_ECG]' + nn_name + '.npy'

#creating directories to save numpy values and the inference plots
dir_csv = os.path.join(dir_root,"csv")
dir_inf_plots = os.path.join(dir_root,"inf_plots")
dir_mae = os.path.join(dir_root,"mae")

m.create_dirs(dir_csv,"numpy inf results")
m.create_dirs(dir_inf_plots,"inference plots")
m.create_dirs(dir_mae, "hri mae")


#saving a list contianing all the HRI MAE calculations
np.savetxt(os.path.join(dir_mae, "mae.csv"), hri_mae, delimiter=",")
#saving a list contianing info on which HRI MAE calculations match number of heart beats
np.savetxt(os.path.join(dir_mae, "hr_match_list.csv"), hr_match_list, delimiter=",")

global_mae, correct_hr_ratio  = m.get_global_mae(hr_match_list, nn_peaks_list, hr_peaks_list, dir_mae)


#%%#########################################################
# Plotting and saving inference results
############################################################


for index in range(len(ecg_real)):
    row_num = index
    str_inf_name = names_list[index]

    str_name_real = str_inf_name + str_real
    str_name_preds =  str_inf_name + str_preds
    str_plot = str_inf_name + nn_name 


    dir_save_real = os.path.join(dir_csv, str_name_real)
    dir_save_preds = os.path.join(dir_csv, str_name_preds)
    dir_save_plot = os.path.join(dir_inf_plots, str_plot)

    
    label1 = 'Real ECG'
    label2 = 'Estimation'
    label3 = 'NN Peaks'


    # hri_mae = []
    # nn_peaks_list = []
    # hr_peaks_list = []
    # hr_match_list = []
    peaks = (nn_peaks_list[row_num])*sample_rate
    signal = ecg_preds[row_num]/max(np.absolute(ecg_preds[row_num]))
                                    
    plt.figure(0, figsize=(10,6))
    #plotting the normalized data 
    plt.plot(ecg_real[row_num]/max(np.absolute(ecg_real[row_num])), label=label1)
    plt.plot(signal, label=label2)
    plt.vlines(peaks, ymin=0, ymax=1, colors='green', linestyles='dashdot', label = label3)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    if hr_match_list[index] == True:
        plt.title(f'Local MAE={hri_mae[index]}, Group MAE={global_mae},')
    else:
        plt.title(f'Local MAE=N/A, Group MAE={global_mae}')
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.01,f'Filename: {str_inf_name}, NN model: {nn_name}', ha='center', fontsize=8)
    #plt.show()
    
    #saving plots 
    plt.savefig(dir_save_plot+'.svg', format='svg')
    plt.savefig(dir_save_plot+'.png', format='png')
    
    
    #saving inference and real ecg as csv files 
    np.savetxt(dir_save_real.replace(".npy", ".csv"), ecg_real[row_num], delimiter=",")
    np.savetxt(dir_save_preds.replace(".npy", ".csv"), ecg_preds[row_num], delimiter=",")

    sio.savemat(dir_save_plot+'[REAL].mat', {"real_ecg": ecg_real[row_num]})
    sio.savemat(dir_save_plot+'[PRED].mat', {"prediction": ecg_preds[row_num]})

    plt.clf()

# print(f'plots saved at: {dir_save_plot}')
# print(f"group MAE: {global_mae}")
# print(f"% HR match: {correct_hr_ratio}")