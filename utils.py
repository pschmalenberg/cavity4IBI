import os 
# data parameters
data_name = "cavity_data"
test_name = "cavity_data"
sample_rate = 2000
batch_size = 8
peak_threshold = 0.4

# training parameters
model_name = "conv-tasnet"  # must be in ["stft", "unet", "conv-tasnet", "sepformer"]
checkpoint = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k"
learning_rate = 1e-5
num_epochs = 50

# spectrogram
n_fft = 256
hop_length = 128

pc_name = "449443" #"Admin"

#C:\Users\Admin\Desktop\2025 Cavity\2025 Neural Network\ckpt
#directory where checkpoint will be saved 
dir_save = os.path.join("C:\\","Users","Admin",
                        "Desktop","2025 Cavity",
                        "2025 Neural Network","ckpt")


#directory where dataset is stored
#inside this dir there should be a sub-folder called [input]
# and another one called [wide_gaussian]
#dir_dataset = os.path.join("C:\\","Users","Admin","Desktop",
#                                "2025 Cavity","2025 Cavity Data Collection",
#                               "dataset")

#directory for the inference data
#training use : wide_gaussian use: C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\dataset
#for inferecne 
#for gaussian
#dir_dataset = os.path.join(r"C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\inference_dataset\Inference_by_p\P7")
#for ecg
dir_dataset = os.path.join(r'C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\raw_ECG_inference_dataset\Inference_by_p\P12')
######################
#for raw ECG training 
#dir_dataset = os.path.join(r"C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\dataset_raw_ECG")



