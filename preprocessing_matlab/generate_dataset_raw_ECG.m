% Generates dataset based on raw data from Biopac in .txt format
% Var1: time 
% Var2: ECG
% Var3: Heart sounds (HS)
% This script will inspect all .txt files in dir_root, load them
% as a Biopac recording, preprocess heart sounds, and convert ECG to
% Gaussian waveform. The dataset will be saved in dir_save

% Preprocessing methods: 
% Heart sounds: downsample 4k -> 2khz, bandpass filter + wavelet denoise
% ECG: downsample 4k -> 2khz, lowpass filter, peak detection to construt
% Gaussian representation
% all inputs are in the form of 4 sec windows, and timestamp appears in 
% the filename  

% all matlab scripts in /dataset_generation_code/ are important to run this
% dataset generation

clc
clear all
close all

% where biometric data is stored
dir_root = ['C:\Users\Admin\Desktop\2025 Cavity\' ...
            '2025 Cavity Data Collection\' ...
            'all data raw\'];

% where dataset is saved at 
dir_save = ['C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\dataset_raw_ECG\'];

dir_gauss = '[wide_gaussian]\';  %it's wide_guassian but basically it is ECG
dir_input = '[input]\';

%creating all sub-directories if they don't exist already:
create_dir(strcat(dir_save, dir_gauss));
create_dir(strcat(dir_save, dir_input));


%listing all .mat files in sub-dir to plot
list_wav_files = dir(fullfile(dir_root,'*.txt'));

for index = 1:size(list_wav_files, 1)
    

    %displaying which file is being processed
    disp(strcat('processing file:',list_wav_files(index).name));
    disp(strcat('index:',num2str(index)));


    % name of file that is being loaded and pplotted 
    str_data = list_wav_files(index).name;
    str_plot = strrep(str_data, '_', '-');
    str_plot = strrep(str_plot, '.txt', '');
    
    %loading data
    data      = readtable(strcat(dir_root, str_data), NumHeaderLines=15);
    fs_orig   = 4e3;
    fs        = 2e3;
    ECG       = data.Var2/max(abs(data.Var2));
    HS        = data.Var3/max(abs(data.Var3));
    
    % this data is originally collected at 4khz, 
    % so it's converted to 2khz first. Wavelet denoise algorithm
    % didn't work too well at 4khz, but works great at 2 khz
    ECG = resample(ECG, fs,  fs_orig);
    HS = resample(HS, fs,  fs_orig);
    time_axis = (1/fs : 1/fs : size(ECG,1)/fs)';
    
    
    %% PREPROCESSING METHODS 
    
    bp_filt_8_128 = designfilt('bandpassiir', ...
                        'FilterOrder', 8, ...
                        'HalfPowerFrequency1', 8, ...
                        'HalfPowerFrequency2', 128, ...
                        'SampleRate', fs);
    % Apply the filter
    HS_filt = filter(bp_filt_8_128, HS);
    % wavelet denoise algorithm 
    HS_filt = wt_noise_reduction(HS_filt, fs);
    % normalizing 
    HS_filt = HS_filt / max(abs(HS_filt));
    
    %denoising ECG signal - helps the wavelet denoising process
    %and the follow up R peak detection 
    Fc = 25; %cutoff frequency 
    % Normalize the cutoff frequency
    Wn = Fc / (fs/2);    % Normalize to Nyquist frequency
    % Design a 4th-order Butterworth low-pass filter
    [b, a] = butter(4, Wn, 'low');
    % Apply the filter
    ECG = filtfilt(b, a, ECG);
   
    % acquiring total file size, in seconds
    % 6 seconds need to be removed because the ECG preprocessing samples
    % 6sec of data at a time, so without the -6 the script goes over the
    % array limits
    total_size = size(ECG,1)/fs;
    total_size = total_size - 6;

    %% GENERATING TRAINING DATA
    %first 5sec of data are ignored
    for xlim_start = 5 : 0.3 : total_size*.8

        %prefix for training 
        prefix = '[A]';

        % each sample is 4 sec of data
        xlim_end = xlim_start + 4;
        
     
        ECG_4s  = ECG(floor(fs*xlim_start) : floor(fs*xlim_end));
        ECG_4s = ECG_4s / max(abs(ECG_4s));

        HS_4s = HS_filt(floor(fs*xlim_start) : floor(fs*xlim_end));
        HS_4s = HS_4s / max(abs(HS_4s));

        


        %% Saving files    

        %sample name starts with the original filename, minus the '[ECG+HS]'
        %suffix
        str_dataset_id   = strrep(str_plot, '[ECG+HS]', '');

        %converting timestamp where the window starts, from float to string
        %and replacing the dot for a comma in the number 
        str_timestamp    = strrep(num2str(xlim_start,'%.1f'), '.',',');

        %defining full sample name - helps tracing back to original files if
        %there's an issue
        str_sample_name  = strcat(prefix,str_dataset_id,'[',str_timestamp,'sec]');

        %full path for preprocessed heart sounds
        dir_hs = strcat(dir_save, dir_input, str_sample_name,'.wav');

        %saving heart sounds as wav
        audiowrite(dir_hs, HS_4s, fs);

        %full path for gaussian ground truth
        dir_gaussian = strcat(dir_save, dir_gauss, str_sample_name);

        %saving gaussian as csv file 
        save_ground_truth(dir_gaussian, ECG_4s);

    end


    %% GENERATING TESTING DATA 
    for xlim_start = (total_size*.8+4) : 0.3 : total_size*.9

        %these files are extra files from P01, and they will all go to
        %training. This happens bc there's far more data from P01 than
        %others. To prevent biases in detection, the excess P01 data goes
        %to training, so testing and validation contains same ratio of data
        %from all participants
        if contains(str_plot, '[P01EXTRA]')
            %training prefix
            prefix = '[A]';
        else
            %prefix for testing
            prefix = '[B1]';
        end

        % each sample is 4 sec of data
        xlim_end = xlim_start + 4;

        ECG_4s  = ECG(floor(fs*xlim_start) : floor(fs*xlim_end));
        ECG_4s = ECG_4s / max(abs(ECG_4s));

        HS_4s = HS_filt(floor(fs*xlim_start) : floor(fs*xlim_end));
        HS_4s = HS_4s / max(abs(HS_4s));



        %% Saving files    

        %sample name starts with the original filename, minus the '[ECG+HS]'
        %suffix
        str_dataset_id   = strrep(str_plot, '[ECG+HS]', '');

        %converting timestamp where the window starts, from float to string
        %and replacing the dot for a comma in the number 
        str_timestamp    = strrep(num2str(xlim_start,'%.1f'), '.',',');

        %defining full sample name - helps tracing back to original files if
        %there's an issue
        str_sample_name  = strcat(prefix,str_dataset_id,'[',str_timestamp,'sec]');

        %full path for preprocessed heart sounds
        dir_hs = strcat(dir_save, dir_input, str_sample_name,'.wav');

        %saving heart sounds as wav
        audiowrite(dir_hs, HS_4s, fs);

        %full path for gaussian ground truth
        dir_gaussian = strcat(dir_save, dir_gauss, str_sample_name);

        %saving gaussian as csv file 
        save_ground_truth(dir_gaussian, ECG_4s);

    end

    %% GENERATING VALIDATION DATA 
    for xlim_start = (total_size*.9+4) : 2 : (total_size-6)
    
    %these files are extra files from P01, and they will all go to training
    if contains(str_plot, '[P01EXTRA]')
        %training
        prefix = '[A]';
    else
        %prefix for validation
        prefix = '[B2]';
    end

    % each sample is 4 sec of data
    xlim_end = xlim_start + 4;
    
    ECG_4s  = ECG(floor(fs*xlim_start) : floor(fs*xlim_end));
    ECG_4s = ECG_4s / max(abs(ECG_4s));
    
    HS_4s = HS_filt(floor(fs*xlim_start) : floor(fs*xlim_end));
    HS_4s = HS_4s / max(abs(HS_4s));

   
    
    %% Saving files    

    %sample name starts with the original filename, minus the '[ECG+HS]'
    %suffix
    str_dataset_id   = strrep(str_plot, '[ECG+HS]', '');
    
    %converting timestamp where the window starts, from float to string
    %and replacing the dot for a comma in the number 
    str_timestamp    = strrep(num2str(xlim_start,'%.1f'), '.',',');
    
    %defining full sample name - helps tracing back to original files if
    %there's an issue
    str_sample_name  = strcat(prefix,str_dataset_id,'[',str_timestamp,'sec]');

    %full path for preprocessed heart sounds
    dir_hs = strcat(dir_save, dir_input, str_sample_name,'.wav');

    %saving heart sounds as wav
    audiowrite(dir_hs, HS_4s, fs);
    
    %full path for gaussian ground truth
    dir_gaussian = strcat(dir_save, dir_gauss, str_sample_name);

    %saving gaussian as csv file 
    save_ground_truth(dir_gaussian, ECG_4s);

    end

end


%% Lots of old plots utilized to check if dataset generation is correct

%time_4sec = (1/fs : 1/fs : size(ecg_4sec,1)/fs)';

% linewidth = 2.5;
% font_size = 14;
% % Plot original and filtered signals
% t = (0:length(ECG)-1)/fs;
% figure;
% plot(t, ECG, 'b', 'DisplayName', 'Original ECG');
% hold on;
% plot(t, ecg_filtered, 'r', 'DisplayName', 'Filtered ECG');
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('ECG Signal - Original vs. 25 Hz Low-pass Filtered');
% legend;
% grid on;

    
% 
% %% Plotting ECG + Detected ECG peak locations
% figure
% 
% plot(time_4sec, ecg_4sec, 'DisplayName', 'ECG', LineWidth=linewidth)
% hold on 
% plot(time_4sec, ecg_gaussian, 'DisplayName', 'Gaussian', LineWidth=linewidth)
% plot(time_4sec, HS_4s, 'DisplayName', 'HS', LineWidth=linewidth)
% xlabel('Time(sec)', 'FontSize', font_size)
% ylabel('Amplitude', 'FontSize', font_size)
% xlim([0 4])
% title(num2str(xlim_start))
% % using for loop to plot position of ECG R peaks 
% for i = 1:length(hr_pos)
%     if i == 1
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'DisplayName', 'ECG Peaks'); 
%     else 
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'HandleVisibility','off'); 
%     end
% end
% set(gca, 'FontSize', font_size);
% legend('Location', 'northoutside', 'Orientation','horizontal')
% hold off
% 
% %% Plotting Gaussian waveform + ECG peak location
% figure
% plot(time_4sec, gaussian_wide, 'DisplayName', 'Gaussian Ground Truth')
% hold on 
% xlabel('Time(sec)', 'FontSize', font_size)
% ylabel('Amplitude', 'FontSize', font_size)
% xlim([0 4])
% % using for loop to plot position of ECG R peaks 
% for i = 1:length(hr_pos)
%     if i == 1
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'DisplayName', 'ECG Peaks'); 
%     else 
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'HandleVisibility','off'); 
%     end
% end
% set(gca, 'FontSize', font_size);
% legend('Location', 'northoutside', 'Orientation','horizontal')
% hold off
% % 
% % % %saving
% % % plot_name = strcat('[GAUSSIAN]',str_plot);
% % % save_fig_name = strcat( dir_save, plot_name);
% % % savefig(strcat(save_fig_name, '.fig'));
% % % print(strcat(save_fig_name, '.png'), '-dpng', '-r300');
% 
% 
% %% Plotting Denoised Heart Sounds + ECG peak location
% figure
% plot(time_4sec, HS_4s, 'DisplayName', 'Preprocessed Heart Sounds')
% hold on 
% plot(time_4sec, ecg_4sec)
% plot(time_4sec, gaussian_wide)
% set(gca, 'FontSize', font_size);
% xlim([0 4])
% 
% % using for loop to plot position of ECG R peaks 
% for i = 1:length(hr_pos)
%     if i == 1
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'DisplayName', 'ECG Peaks'); 
%     else 
%         xline((hr_pos(i)), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'HandleVisibility','off'); 
%     end
% end
% legend('Location', 'northoutside', 'Orientation','horizontal')
% hold off
% % 
% %saving
% % plot_name = strcat('[HS_PREPROCESSED]',str_plot);
% % save_fig_name = strcat( dir_save, plot_name);
% % savefig(strcat(save_fig_name, '.fig'));
% % print(strcat(save_fig_name, '.png'), '-dpng', '-r300');
% 
% 
% %% Plot Unprocessed Heart sounds + ECG peak location
% figure
% plot(time_axis, HS, 'DisplayName', 'Original Heart Sounds')
% hold on 
% set(gca, 'FontSize', font_size);
% xlim([xlim_start xlim_end])
% 
% % using for loop to plot position of ECG R peaks 
% for i = 1:length(hr_pos)
%     if i == 1
%         xline((hr_pos(i)+xlim_start), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'DisplayName', 'ECG Peaks'); 
%     else 
%         xline((hr_pos(i)+xlim_start), 'Color', "#D95319", 'LineStyle', ':', ...
%             "LineWidth", linewidth,'HandleVisibility','off'); 
%     end
% end
% legend('Location', 'northoutside', 'Orientation','horizontal')
% hold off
% 
% %saving
% % plot_name = strcat('[HS_RAW]',str_plot);
% % save_fig_name = strcat( dir_save, plot_name);
% % savefig(strcat(save_fig_name, '.fig'));
% % print(strcat(save_fig_name, '.png'), '-dpng', '-r300');
    