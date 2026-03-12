import pandas as pd
import os 
import matplotlib.pyplot as plt 
import numpy as np
import torch 
import pickle
import re
from numpy import genfromtxt
import math
import csv 
import methods as m
from utils import *
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from dtw import dtw as dtw_func
from scipy.ndimage import zoom


# 8.X series: running on different ground truths after 6.7D1 parameters were identified as best for wide gaussian
# 8,3b [2025-02-14][17h38min02sec] *** raw ecg ***
# 8,3a [2025-02-13][23h00min18sec] *** raw ecg ***
# 8,2c [2025-02-12][17h32min04sec] *** r-wave only ***
# 8,2b2 [2025-02-12][10h34min51sec]*** r-wave only ***
# 8,2b [2025-02-12][05h44min21sec] *** r-wave only ***
# 8,1b [2025-02-07][12h24min18sec] *** narrow gaussian ***
# 8,1a [2025-02-07][05h01min46sec] *** narrow gaussian ***
# 7.1B: [2025-02-03][09h13min12sec] (final checkpoint)
# 7.1A: [2025-01-31][14h11min47sec]
# 6.7E: [2025-01-30][01h39min49sec]
# 6.7D2 [2025-01-29][10h20min01sec]
# 6.7D1 [2025-01-28][02h18min58sec]
# 6.7A: [2025-01-23][13h19min12sec]
# 6.4D: [2025-01-14][08h34min24sec]
# 6.4C: [2025-01-09][15h14min16sec]
# 5.7B: [2024-12-09][10h17min20sec]ca-dense-unet-cnn_200_epochs
# 5.4:  [2024-11-25][12h22min13sec]ca-dense-unet-cnn_200_epochs
# 5.3:  [2024-11-21][11h21min35sec]ca-dense-unet-cnn_200_epochs
# 5.2:  [2024-11-15][09h41min19sec]ca-dense-unet-cnn_200_epochs
# 4.5B: [45B][2024-04-25][22h49min40sec]drive_ca-dense-unet-cnn_300_epochs


#saving text is off at the bottom of the code 
def saving_text(global_mae_3, global_hr_percent_3, hr_real_global, hr_pred_global,dir_cont_data):

    #saving data 
    text_content = f"""
    Global MAE: {global_mae_3}
    Global HR: {round(global_hr_percent_3*100,2)}%
    Ground truth HR: {round(hr_real_global,1)}bpm
    PREDICTION HR: {round(hr_pred_global,1)}bpm
    """

    text_name = os.path.join(dir_cont_data, "global_HRI_and_HR.txt")

    #saving text
    with open(text_name, 'w') as file: 
        file.write(text_content)
#plotting and saving plots is off in this method and in the code below
def plot_continuous_data(array_real, array_pred, duration, start_time):
    end_time = start_time + duration
    time_axis = np.arange(start_time, end_time, 1/sample_rate)

    plt.figure(figsize=(20,4))

    #plotting the normalized data 
    plt.plot(array_real[int(start_time*sample_rate) : int(end_time*sample_rate)], label = 'real ecg')  #time_axis, 
    plt.plot( array_pred[int(start_time*sample_rate) : int(end_time*sample_rate)], label = 'prediction')  #time_axis,
    plt.xlabel('Time[sec]')
    plt.ylabel('Amplitude')
    #if array_real.shape[0] > 790000:
    str_filename = 'saved_files' #"[2024_10_02][4][2831-3515][100kmph][B2]"
    # else: 
    #     str_filename = "[2024_10_02][4][3515-3747][100kmph][B2]"
    plt.figtext(0.5, -0.05,f'Filename: {str_filename}, NN model: {nn_name_full}', ha='center', fontsize=10, color='gray')
    plt.legend()
    plt.grid(True)
    plot_name = '[' + str(duration) + ']' + '[' + str(start_time) + ']' + str_filename
    # plt.savefig(os.path.join(dir_cont_data, plot_name + '.svg'), format='svg')
    plt.savefig(os.path.join(dir_cont_data, plot_name + '.png'), format='png')
def get_peaks(array):
    '''
    This function gets the peaks from the array
    - max hr in humans: 220 bpm minus age
    - for an 18 years old: 202bpm = 13.5 beats per 4 secs = 1beat per 296ms
    - ******NEW REF: 208-.7*age = 195 for a 18 years old = 308ms = 208ms/.5ms = 616******
    - source: Age-predicted maximal heart rate revisited https://www.jacc.org/doi/abs/10.1016/S0735-1097(00)01054-8
    - at 2 kHz: 1 px = 0.5ms, so min distance = 296ms/.5ms = 592 px
    '''
    peaks, _ = find_peaks(array, height=peak_threshold, distance = 616)
    return peaks
def get_heartbeat_match(ecg, pred):
    ''' 
    Checks if the heartbeat peaks between ground truth and estiamtion 
    are matching.
    Parameters: 
    - ecg: array containig time location of all ground truth peaks
    - pred: array containig time location of all NN predicted peaks

    Output: 
    - matching_heartbeats: 
        - array of 2 rows and n columns, n being the number of beats in ecg
        - if entries are zero, it means there was no HR match
        - if entries are other number, it shows the match 
    '''
    hr_match = np.zeros(shape=(2, ecg.shape[0]))

    #looping over each element in ground truth,
    #except first and last elements bc of edge issues
    for i in np.arange(1,(ecg.shape[0]-1)):

        #establishing lower and upper boundaries around a heartbeat 
        gap_l = (ecg[i]- ecg[i-1])/2
        gap_r = (ecg[i+1] - ecg[i])/2
        
        #list to store all potential heartbeat candidates from prediction array
        candidates_list = []
        #looping over prediction array, to check if an element fits the gap 
        for j in np.arange(1,(pred.shape[0]-1)):
            if pred[j] > ecg[i]-gap_l and pred[j] < ecg[i]+gap_r:
                candidates_list.append(pred[j])
            
        #if there is one element in candidates_list, this is a correct HR match
        #if there is 0 elements, or >1 element, it means predition failed for that 
        #specific heartbeat
        if len(candidates_list) == 1:
            #print("correct estimation for ecg[i]")
            hr_match[0,i] = ecg[i]
            hr_match[1,i] = candidates_list[0]

    #getting indices of columns containing only zeros (where there was no match)
    zero_cols = np.where(~hr_match.any(axis=0))[0]
    #deleting columns containing only zeros 
    hr_match_final = np.delete(hr_match, zero_cols, axis=1)
    

    return hr_match_final
def get_global_analysis(match_array, total_peaks):
    '''
    This method calculates %HR match and HRI MAE for the entire recording 
    - total_peaks: total number of heartbeats present in ground truth 
        IMPORTANT: first and last ecgpeaks are not analized for hr_match,
        so total peaks is the size of ECG array minus 2
    - match_array contains 2 rows and lists the pairs of pred + real heartbeats 
        this is the output from get_heartbeat_match
    '''

    #percentage of pred hr that match ground truth 
    percent_match = match_array.shape[1]/total_peaks

    # hri mae needs to be <5ms 80% of the time, 
    # so the best 80% of matches will be selected 
    num_beats_for_hri = math.floor(total_peaks*(percent_match-0.0001))  #0.8 

    #getting time difference between pred and ground truth 
    hr_difference = abs(match_array[0] - match_array[1])
    try: 
        best_indexes = np.argpartition(hr_difference, num_beats_for_hri)[:num_beats_for_hri]
        #sorting indexes in ascending order
        best_indexes = np.sort(best_indexes)

        best_match = np.zeros(shape=(2, best_indexes.shape[0]))
        #i is the index containing the best pairs of ecg + pred
        index = 0
        for i in best_indexes:
            #print(i)
            #ecg values
            best_match[0, index] = match_array[0, i]
            #pred values 
            best_match[1, index] = match_array[1, i]

            #updating index 
            index = index+1

        ecg_hri  = (best_match[0,1:] - best_match[0,:-1]) / sample_rate
        pred_hri = (best_match[1,1:] - best_match[1,:-1]) / sample_rate

        #hri mae for the best 80% fo all the heartbeats
        hri_mae = np.mean(np.abs(ecg_hri - pred_hri))
    except: 
        hri_mae = 'NaN'



    return percent_match, hri_mae
def sort_by_bracket_number(string_list):
  """
  This function will rearrange the list elements chronologically. 
  In each list, the files are not ordered in chronological order. 
  However, each filname contains a timestamp in brackets. 

  Args:
    string_list: The list of strings to be sorted.

  Returns:
    A new list with the strings sorted by the number within the brackets.
  """

  def extract_number(string):
    """Extracts the number within brackets from a string."""
    #re.search(pattern, string): This is a function from the re module used for pattern matching in strings.
    #pattern: This is the regular expression that defines the pattern to search for.
    #string: This is the string where the search will be conducted.
    string = string.replace(",", "]")
    match = re.search(r"\[(\d{1,5})\]", string)
    return int(match.group(1)) if match else 0

  return sorted(string_list, key=extract_number)
def get_continuous_data(list, extension):
    '''
    this method will load the data files based on their names from list element
    then the files will be concatenated and averaged, to generate one continuous file 
    that contains all elements from the list
    - list: this list contains filenames sorted chronologically, each file is 4 sec long data with 50% overlap with the consecutive file.  
    - extension: it's a string, either [PRED_ECG] or [REAL_ECG].
    '''
    
    array = np.zeros((2*len(list)+8)*sample_rate)

    lim_low = 0
    lim_up = int(lim_low + 4*sample_rate)
    for data_name in list: 
        path = os.path.join(dir_root, nn_name_full, "csv", data_name + extension + nn_name_full+".csv")
        data = genfromtxt(path,delimiter=' ')
        array[lim_low:lim_up] = array[lim_low:lim_up] + data

        #updating indices
        lim_low = int(lim_low + 2*sample_rate)
        lim_up  = int(lim_low + 4*sample_rate)

    #normalizing the array: overlap doesn't happen in first 2sec of data, 
    #or in the last 2 sec. So before normalizing, these need to be corrected 
    array[0:2*sample_rate] = 2*array[0:2*sample_rate] 

    #removing trailing zeros at the end of the array
    array = array[0:(lim_up-4*sample_rate)]

    #normalizing
    array = array / np.max(np.abs(array))

    return array 

# dir_root     = os.path.join("C:\\", "Users","Admin", "Desktop", "hr_hri",
#                             "wav2ecg", "inference_tests")

dir_root    = dir_save

#listing all directories
nn_name_full_list = []

# for item in os.listdir(dir_root):
#     if "[2025-02-07]" in item or "[2025-02-08]" in item:# or "[2025-02-15]" in item:
#         nn_name_full_list.append(item)

#nn_name_full_list.append('2025-10-17_16h30min')
nn_name_full_list.append('2025-11-22_23h02min')


for nn_name_full in nn_name_full_list:
    #nn_name_full = m.get_latest_subfolder(dir_root)
    dir_names    = os.path.join(dir_root, nn_name_full, "npy", "[NAMES_INF]" + nn_name_full + ".pkl")
    dir_preds    = os.path.join(dir_root, nn_name_full, "npy", "[PRED_ECG]" + nn_name_full + ".npy")
    dir_real     = os.path.join(dir_root, nn_name_full, "npy", "[REAL_ECG]" + nn_name_full + ".npy")


    preds = np.load(dir_preds)
    reals = np.load(dir_real)


    #loading the list containing names of all 4sec dataset files used for inference 
    with open(dir_names, 'rb') as file:
        NAMES_INF = pickle.load(file)

    #print(NAMES_INF) 

    '''
    All 100kmph data used in inference comes from the file [2024_10_02][4], shared by MIRISE.
    Two moments of the dataset were used for 100kmph extraction.
    Full NAMES_INF contains 4sec snippets data from these two different sources: 
    1. a 100kmph drive that starts in time = 2831 sec and goes until time = 3515sec
        - files from this source have a tag "[2831-3515]" on filename
    2. a 100kmph drive that starts in time = 3515 sec and goes until time = 3747sec
        - files from this source have a tag "[3515-3747]" on filename

    There is a 50% overlap between two consecutive files
        
    These two sources will be separated into individual lists
    '''

    list_2831 = NAMES_INF # [s for s in NAMES_INF if "[2831-3515]" in s]
    # list_3515  = [s for s in NAMES_INF if "[3515-3747]" in s]

    # #verifying the list is accurate:
    # print("list_2831: ")
    # for names in list_2831:
    #     print(names)

    # #verifying the list is accurate:
    # print("list_3513: ")
    # for names in list_3515:
    #     print(names)



    sorted_2831 = sort_by_bracket_number(list_2831)
    # sorted_3515 = sort_by_bracket_number(list_3515)


    # #verifying accuracy of sorting process: 
    # for names in sorted_3515:
    #     print(names)


    list_subset = sorted_2831[0:30]

    # array_pred = get_continuous_data(list, "[PRED_ECG]")
    # array_real = get_continuous_data(list, "[REAL_ECG]")


    # len(array_pred)/sample_rate
    continuous_pred_2831 = get_continuous_data(sorted_2831, "[PRED_ECG]")
    continuous_real_2831 = get_continuous_data(sorted_2831, "[REAL_ECG]")

    # continuous_pred_3515 = get_continuous_data(sorted_3515, "[PRED_ECG]")
    # continuous_real_3515 = get_continuous_data(sorted_3515, "[REAL_ECG]")

    #print(f"2831 continuous data: {continuous_pred_2831.shape[0]/sample_rate} sec long")
    #print(f"3515 continuous data: {continuous_pred_3515.shape[0]/sample_rate} sec long")


    dir_cont_data = os.path.join(dir_root, nn_name_full, "continuous_data")
    #checking if a dir exists. if not, create it 
    if not os.path.exists(dir_cont_data):
        os.makedirs(dir_cont_data)

    str_filename = 'continuous_pred_2831'
    np.savetxt(os.path.join(dir_cont_data, str_filename + '.csv'), continuous_pred_2831, delimiter=',', fmt='%f')

    str_filename = 'continuous_real_2831'
    np.savetxt(os.path.join(dir_cont_data, str_filename + '.csv'), continuous_real_2831, delimiter=',', fmt='%f')

    # str_filename = 'continuous_pred_3515'
    # np.savetxt(os.path.join(dir_cont_data, str_filename + '.csv'), continuous_pred_3515, delimiter=',', fmt='%f')

    # str_filename = 'continuous_real_3515'
    # np.savetxt(os.path.join(dir_cont_data, str_filename + '.csv'), continuous_real_3515, delimiter=',', fmt='%f')




    start = np.arange(0, int(continuous_real_2831.shape[0]/sample_rate), 30)
    for p in start:
        plot_continuous_data(continuous_real_2831, continuous_pred_2831, duration=30, start_time=p)
        plt.close('all')


    peak_threshold = 0.4



    # peaks_real_3515 = get_peaks(continuous_real_3515)
    # peaks_pred_3515 = get_peaks(continuous_pred_3515)

    peaks_real_2831 = get_peaks(continuous_real_2831)
    peaks_pred_2831 = get_peaks(continuous_pred_2831)

    match_2831 = get_heartbeat_match(ecg=peaks_real_2831, pred=peaks_pred_2831)
    # match_3515 = get_heartbeat_match(ecg=peaks_real_3515, pred=peaks_pred_3515)



    percent_match_2831, hri_mae_2831 = get_global_analysis(match_2831, (peaks_real_2831.shape[0]-2))
    # percent_match_3515, hri_mae_3515 = get_global_analysis(match_3515, (peaks_real_3515.shape[0]-2))

    # print("_2831 recording: ")
    # print(f"% of hr match: {percent_match_2831}")
    # print(f"hri_mae: {hri_mae_2831}")
    # print("\n")
    # print(f"_3515 recording: ")
    # print(f"% of hr match: {percent_match_3515}")
    # print(f"hri_mae: {hri_mae_3515}")

    # Pearson Correlation between predicted and real ECG

    pearson_corr, pearson_pval = pearsonr(continuous_real_2831, continuous_pred_2831)
    print(f"\n===== Signal Similarity Metrics =====")
    print(f"Pearson Correlation: {round(pearson_corr, 4)} (p-value: {pearson_pval:.2e})")

    # Dynamic Time Warping (DTW) between predicted and real ECG
    # Downsample for DTW computation efficiency (DTW is O(n^2) in memory/time)
    downsample_factor = 10
    real_ds = continuous_real_2831[::downsample_factor]
    pred_ds = continuous_pred_2831[::downsample_factor]

    dtw_result = dtw_func(real_ds, pred_ds)
    dtw_distance = dtw_result.distance
    dtw_normalized = dtw_distance / len(real_ds)
    print(f"DTW Distance: {round(dtw_distance, 4)}")
    print(f"DTW Normalized Distance: {round(dtw_normalized, 6)}")
    print(f"=====================================\n")

    # Save similarity metrics to CSV
    similarity_csv_path = os.path.join(dir_cont_data, "similarity_metrics.csv")
    with open(similarity_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Pearson Correlation", round(pearson_corr, 4)])
        writer.writerow(["Pearson p-value", pearson_pval])
        writer.writerow(["DTW Distance", round(dtw_distance, 4)])
        writer.writerow(["DTW Normalized Distance", round(dtw_normalized, 6)])
        writer.writerow(["Downsample Factor (DTW)", downsample_factor])

    # ======== Grad-CAM Visualization on Predicted ECG ========
    # Highlight which temporal features of the input contribute most to the prediction

    try:
        from models import ConvTasNet  # import model architecture

        # Try two possible checkpoint paths:
        # 1) subfolder structure: ckpt/{nn_name}/ckpt/{nn_name}.pth  (from inference_mc_mae.py)
        # 2) flat structure:      ckpt/{nn_name}.pt                  (from inference.py)
        checkpoint_path_pth = os.path.join(dir_save, nn_name_full, "ckpt", nn_name_full + ".pth")
        checkpoint_path_pt  = os.path.join(dir_save, nn_name_full + ".pt")

        if os.path.exists(checkpoint_path_pth):
            checkpoint_path = checkpoint_path_pth
        elif os.path.exists(checkpoint_path_pt):
            checkpoint_path = checkpoint_path_pt
        else:
            checkpoint_path = None

        if checkpoint_path is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Instantiate the model architecture, then load saved weights
            model = ConvTasNet()
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Select a representative 4-second segment from the middle of the recording
            seg_start = len(continuous_real_2831) // 2
            seg_end = seg_start + int(4 * sample_rate)
            if seg_end > len(continuous_real_2831):
                seg_start = 0
                seg_end = int(4 * sample_rate)

            input_segment = continuous_real_2831[seg_start:seg_end]
            # ConvTasNet expects (batch, samples) — 2D input, NOT (batch, 1, samples)
            input_tensor = torch.tensor(input_segment, dtype=torch.float32).unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # Hook to capture intermediate activations and gradients from the last conv layer
            activations = {}
            gradients = {}

            def forward_hook(module, input, output):
                activations['value'] = output

            def backward_hook(module, grad_input, grad_output):
                gradients['value'] = grad_output[0]

            # Find the last convolutional layer in the model
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    target_layer = module
                    target_layer_name = name

            if target_layer is not None:
                handle_fwd = target_layer.register_forward_hook(forward_hook)
                handle_bwd = target_layer.register_full_backward_hook(backward_hook)

                # Forward pass
                output = model(input_tensor)

                # ConvTasNet returns a list of tensors (one per speaker)
                if isinstance(output, (list, tuple)):
                    output = output[0]

                # Backward pass: use sum of output as scalar target
                model.zero_grad()
                output.sum().backward()

                # Compute Grad-CAM
                grads = gradients['value']
                acts = activations['value']

                weights = torch.mean(grads, dim=-1, keepdim=True)
                grad_cam = torch.sum(weights * acts, dim=1).squeeze()
                grad_cam = torch.relu(grad_cam)
                grad_cam = grad_cam.detach().cpu().numpy()

                # Upsample Grad-CAM to match original signal length
                grad_cam_upsampled = zoom(grad_cam, int(4 * sample_rate) / len(grad_cam))
                grad_cam_upsampled = grad_cam_upsampled / (np.max(grad_cam_upsampled) + 1e-8)

                pred_segment = continuous_pred_2831[seg_start:seg_end]
                real_segment = continuous_real_2831[seg_start:seg_end]
                time_axis_seg = np.arange(len(pred_segment)) / sample_rate

                # Set publication-quality font sizes for Nature
                plt.rcParams.update({
                    'font.size': 18,
                    'axes.titlesize': 22,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16,
                    'legend.fontsize': 16,
                    'figure.titlesize': 24,
                    'font.family': 'Arial',
                })

                # Plot 1: Grad-CAM overlay on predicted ECG
                fig1, ax1 = plt.subplots(1, 1, figsize=(20, 5))
                ax1.plot(time_axis_seg, pred_segment, color='blue', linewidth=1.2, label='Predicted ECG')
                ax1.fill_between(time_axis_seg, pred_segment.min(), pred_segment.max(),
                                    where=grad_cam_upsampled[:len(time_axis_seg)] > 0.3,
                                    alpha=0.3, color='red', label='High Grad-CAM activation')
                im = ax1.scatter(time_axis_seg, pred_segment, c=grad_cam_upsampled[:len(time_axis_seg)],
                                    cmap='jet', s=2, zorder=5)
                ax1.set_xlabel('Time [sec]', fontsize=20, fontweight='bold')
                ax1.set_ylabel('Amplitude', fontsize=20, fontweight='bold')
                ax1.set_title(f'Grad-CAM on Predicted ECG (layer: {target_layer_name})', fontsize=22, fontweight='bold')
                ax1.legend(loc='upper right', fontsize=16, framealpha=0.9)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='both', which='major', labelsize=16)
                cbar = plt.colorbar(im, ax=ax1, label='Grad-CAM Importance')
                cbar.ax.tick_params(labelsize=14)
                cbar.set_label('Grad-CAM Importance', fontsize=18, fontweight='bold')
                plt.tight_layout()
                gradcam_plot1_name = os.path.join(dir_cont_data, 'grad_cam_overlay.png')
                plt.savefig(gradcam_plot1_name, format='png', dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(dir_cont_data, 'grad_cam_overlay.svg'), format='svg', bbox_inches='tight')
                print(f"Grad-CAM overlay plot saved to: {gradcam_plot1_name}")
                plt.close('all')

                # Plot 2: Grad-CAM activation map vs Real & Predicted ECG
                fig2, ax2 = plt.subplots(1, 1, figsize=(20, 5))
                ax2.plot(time_axis_seg, real_segment, color='green', linewidth=1.2, alpha=0.7, label='Real ECG')
                ax2.plot(time_axis_seg, pred_segment, color='blue', linewidth=1.2, alpha=0.7, label='Predicted ECG')
                ax2.bar(time_axis_seg, grad_cam_upsampled[:len(time_axis_seg)] * np.max(np.abs(pred_segment)),
                            width=1/sample_rate, alpha=0.3, color='red', label='Grad-CAM weight')
                ax2.set_xlabel('Time [sec]', fontsize=20, fontweight='bold')
                ax2.set_ylabel('Amplitude / Importance', fontsize=20, fontweight='bold')
                ax2.set_title('Grad-CAM Activation Map vs Real & Predicted ECG', fontsize=22, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=16, framealpha=0.9)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='both', which='major', labelsize=16)
                plt.tight_layout()
                gradcam_plot2_name = os.path.join(dir_cont_data, 'grad_cam_activation_map.png')
                # plt.savefig(gradcam_plot2_name, format='png', dpi=300, bbox_inches='tight')
                # plt.savefig(os.path.join(dir_cont_data, 'grad_cam_activation_map.svg'), format='svg', bbox_inches='tight')
                print(f"Grad-CAM activation map plot saved to: {gradcam_plot2_name}")
                plt.close('all')

                # Reset rcParams to defaults
                plt.rcParams.update(plt.rcParamsDefault)

                handle_fwd.remove()
                handle_bwd.remove()
            else:
                print("Grad-CAM: No convolutional layer found in the model.")
        else:
            print(f"Grad-CAM: Checkpoint not found at {checkpoint_path}. Skipping.")
    except Exception as e:
        print(f"Grad-CAM visualization failed: {e}")
        print("Skipping Grad-CAM analysis.")



    try:
        test = peaks_pred_2831[-2]
        #total time in first recording is time between second heartbeat to second to last heartbeat (first and last heartbeats are never analyzed)
        time_2831 = (peaks_pred_2831[-2] - peaks_pred_2831[1]) / sample_rate

        #heart rate per second: num_beats / time_sec
        #heart rate per minute: num_beats * 60 / time_sec 
        hr_real_2831 = (peaks_real_2831.shape[0]-2)*60 / time_2831
        hr_pred_2831 = (peaks_pred_2831.shape[0]-2)*60 / time_2831

        # print("Heart rate for 2831 recordings: ")
        # print(f"ground truth HR: {round(hr_real_2831,1)}bpm")
        # print(f"pred truth HR: {round(hr_pred_2831,1)}bpm")

    except IndexError: 
        hr_real_2831 = 'NaN'
        hr_pred_2831 = 'NaN'

        # print("Heart rate for 2831 recordings: ")
        # print(f"ground truth HR: {hr_real_2831,1}bpm")
        # print(f"pred truth HR: {hr_pred_2831,1}bpm")



    # try:
    #     #total time in first recording is time between second heartbeat to second to last heartbeat (first and last heartbeats are never analyzed)
    #     time_3515 = (peaks_pred_3515[-2] - peaks_pred_3515[1]) / sample_rate

    #     #heart rate per second: num_beats / time_sec
    #     #heart rate per minute: num_beats * 60 / time_sec 
    #     hr_real_3515 = (peaks_real_3515.shape[0]-2)*60 / time_3515
    #     hr_pred_3515 = (peaks_pred_3515.shape[0]-2)*60 / time_3515

    #     # print("Heart rate for 3515 recordings: ")
    #     # print(f"ground truth HR: {round(hr_real_3515,1)}bpm")
    #     # print(f"pred truth HR: {round(hr_pred_3515,1)}bpm")
    # except IndexError:
    #     hr_real_3515 = 'NaN'
    #     hr_pred_3515 = 'NaN'
    #     # print("Heart rate for 3515 recordings: ")
    #     # print(f"ground truth HR: {hr_real_3515,1}bpm")
    #     # print(f"pred truth HR: {hr_pred_3515,1}bpm")

    #global heart rate 
    try:
        #calculating global heart rate: total num heartbeats divided by total time
        total_time =  time_2831  #time_3515 +
        hr_real_global = ( peaks_real_2831.shape[0]-2)*60 / total_time    #peaks_real_3515.shape[0]-2 +
        hr_pred_global = ( peaks_pred_2831.shape[0]-2)*60 / total_time    #peaks_pred_3515.shape[0]-2 +

        
        print("==============Global heart rate analysis (2831 + 3515):=============== ")
        print("====================================================================== ")
        print(f"NN version: {nn_name_full}")
        print(f"ground truth HR: {round(hr_real_global,1)}bpm")
        print(f"PREDICTION HR: {round(hr_pred_global,1)}bpm")



        #calculating %HR and HRI MAE: 
        match_total = match_2831# np.concatenate((match_2831), axis=1)   #,match_3515
        peaks_real_total =  (peaks_real_2831.shape[0]-2)  #(peaks_real_3515.shape[0]-2) +
        global_hr_percent_3, global_mae_3 = get_global_analysis(match_total, peaks_real_total)

        try: 
            print(f"global HRI mae: {round(global_mae_3*1000,2)}ms")
        except: 
            print(f"global HRI mae: {global_mae_3}")
        print(f"global HR match: {round(global_hr_percent_3*100,2)}%")

    except: 
        global_mae_3 = 'NaN'
        global_hr_percent_3 = 0
        print('not enough HR detected to calculate anything')
    
    saving_text(global_mae_3, global_hr_percent_3, hr_real_global, hr_pred_global,dir_cont_data)