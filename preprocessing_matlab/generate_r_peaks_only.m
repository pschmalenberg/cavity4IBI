function [r_waveform] = generate_r_peaks_only(fs, ECG, R_peak_pos)

window = 0.06; %R-peak lasts 35ms on average: 
               %R-Peak Time: An Electrocardiographic Parameter with Multiple Clinical Applications 

time = 1/fs : 1/fs : size(ECG,1)/fs;
time = time';
mask = zeros(size(time,1), 1);

%assigning mask=1 in a 60ms window around each R peak position
for j=1:size(R_peak_pos,1)
    for i=1:size(time,1)
        lim_upper = R_peak_pos(j)+window/2;
        lim_lower = R_peak_pos(j)-window/2;
        if time(i) > lim_lower && time(i) < lim_upper
            mask(i)=1;
        end
    end
end 




r_waveform = mask.*ECG;

% removing negative components 
r_waveform(r_waveform<0) = 0;

%checking if R-peaks are flipped - if they are, integral of signal is neg
%this happens if sensors are flipped when recording ECG data
if sum(r_waveform) < 0
    r_waveform = r_waveform.*(-1);
end

% %removing negative components 
% for i=1:size(r_waveform, 1)
%     if r_waveform(i) < 0 
%         r_waveform(i) = 0;
%     end 
% end


%normalizing 
r_waveform = r_waveform/max(r_waveform);

% figure
% plot(time, ECG, time, r_waveform)
% xlim([0 4])

end