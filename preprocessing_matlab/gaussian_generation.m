%input: array of 6sec ecg data and the sampling frequency 
%output: 4 different waveforms related to heart data:
%ecg raw, r peaks only, narrow gaussian, wide gaussian
%first and last second of data is removed from output, so final output is
%4sec long

function [ecg_4sec, ecg_r, gaussian_wide, hr_pos]= gaussian_generation(ecg_6s, fs)


%%%%%%%%%%%%%%%%%%%
%extracting R peaks
%%%%%%%%%%%%%%%%%%%


%1. performs wavelet transform on ECG signal
%modwt(signal,int) generates int rows, each one containing one wavelet
%transform order
%modwt = maximal overlap discrete wavelet transform
%to enhance the R peaks in the ECG waveform
wt = modwt(ecg_6s,5);

%2. getting only scales 4 and 5 of the previous trasform
%scales 4 and 5 correspond to relevant frequency bands for ECG:
%Scale 4 -- [11.25, 22.5) Hz, Scale 5 -- [5.625, 11.25) Hz.
wtrec = zeros(size(wt));
wtrec(4:5,:) = wt(4:5,:);


%replacing intial .5 sec of data with the median because there is an
%overshooting pulse at the start of function
wtrec(4,1:(.5*fs)) = median(wtrec(4,:));
wtrec(5,1:(.5*fs)) = median(wtrec(5,:));

%also replacing the last .5sec of the data with median in case there is an
%overshooting pulse at the end of modwt
wtrec(4,(end-.5*fs):end) = median(wtrec(4,:));
wtrec(5,(end-.5*fs):end) = median(wtrec(5,:));

%normalizing the rest of the values bc they are in the order of 1e-8
wtrec(4,:) = wtrec(4,:) / max(wtrec(4,:));
wtrec(5,:) = wtrec(5,:) / max(wtrec(5,:));

%3. reconstructing signal, using only scale 4 and 5 of original wavelet
%transform decomposition
%sym4 is a wavelet that resembles ECG QRS complex
y = (imodwt(wtrec,'sym4'))';

%only working with positive values from y
%in case r peak has a strong negative component, this prevents
%that component from being the peak detected
y_p = y;
y_p(y_p<0)=0;

% 4. squaring the output to enhance peaks
y_norm = abs(y_p).^2;

% normalize signal to avoid vanishing values
y_norm = y_norm / max(y_norm);

%time_axis to get peak position
time_axis = (1/fs : 1/fs : size(y_norm, 1)/fs)';

% 
% [pxx, f] = pwelch(ecg_6s, [], [], [], fs);
% % Plot the power spectrum
% figure;
% plot(f, 10*log10(pxx));
% xlabel('Frequency (Hz)');
% ylabel('Power/Frequency (dB/Hz)');
% title('Power Spectrum of ECG Signal');
% 
% 
% % Compute and plot the spectrogram
% figure;
% spectrogram(ecg_6s, 256, 200, 256, fs, 'yaxis');
% title('Spectrogram of ECG Signal');
% colorbar;
% 
% 
% 

%locating peaks in a 6 seconds window 
[~,R_peak_pos] = findpeaks(y_norm,time_axis,'MinPeakHeight',0.1,...
'MinPeakDistance',0.4);

%generating a waveform that contains only the R-peak component of ECG,
%in a 6 second window 
ecg_r = generate_r_peaks_only(fs, ecg_6s, R_peak_pos);

% figure
% %plot(y/max(abs(y)))
% hold on
% plot(ecg_r)
% % plot(ecg_r/max(abs(ecg_r)))
% plot(ecg_6s/max(abs(ecg_6s)))
% % plot(y_norm)
% hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating gaussian waveform based on R peaks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vector_size = size(ecg_r,1);

[~,peak_pos]  = findpeaks(ecg_r,time_axis,'MinPeakHeight',0.2,'MinPeakDistance',0.3);

gaussian_wide = zeros(size(ecg_r,1),1);

for i=1:size(peak_pos,1)
    center = peak_pos(i)*fs;
    amplitude = 1;
    width = .1*fs;

    % Create a vector of zeros
    gaussian_curve = zeros(vector_size, 1);
    
    % Generate x-axis values for the Gaussian curve
    x_axis = 1:vector_size;
    
    % Calculate the Gaussian distribution using the formula
    sigma = width / sqrt(2*log(2));  % Standard deviation based on width
    gaussian_curve = amplitude * exp(-((x_axis - center).^2) / (2*sigma^2));
    
    %figure; plot(gaussian_curve');

    gaussian_wide = gaussian_wide + gaussian_curve';

end


% clipping signals to 4sec windows 
%[ecg_4sec, ecg_r, gaussian_narrow, gaussian_wide]= r_peaks_4sec_window(ecg_6s, fs)
lim_lower = fs;
lim_upper = size(y,1)-fs-1;

%removing padding from 6sec ECG data
ecg_4sec = ecg_6s(lim_lower:lim_upper,1);
%ecg_r = ecg_r(lim_lower:lim_upper,1);

gaussian_wide = gaussian_wide(lim_lower:lim_upper,1);

%extracting position of hr peaks as the position of gaussian_wide peaks
[~,hr_pos]  = findpeaks(gaussian_wide,(1/fs:1/fs:size(gaussian_wide,1)/fs)','MinPeakHeight',0.2,'MinPeakDistance',0.3);

% % %confirming if peaks in ECG array match real ECG from data
% figure
% plot(gaussian_wide, LineWidth=1.5)
% hold on
% 
% for i=1:size(test2,1)
%     xline(test2(i,1)*fs, 'Color', 'red', 'LineWidth',1.5)
% end
% 
% hold off


