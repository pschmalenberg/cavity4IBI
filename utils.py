import os 

release_root = os.path.dirname(os.path.abspath(__file__))

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

dir_save = os.environ.get("CAVITY_OUTPUT_DIR", os.path.join(release_root, "outputs"))
dir_dataset = os.environ.get(
    "CAVITY_DATASET_DIR", os.path.join(release_root, "data", "processed")
)



