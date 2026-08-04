function []=save_ground_truth(dir_save, ECG)



%saving ECG as CSV format
ECG_CSV = [0; ECG;];

data_size = size(ECG_CSV,1)-1;

data_index = (0:(data_size-1))'; 
data_index = num2str(data_index);
data_index = ["";data_index];
data_index = strtrim(data_index);

data_new = [data_index, num2str(ECG_CSV, '%.4f') ];

writematrix(data_new, strcat(dir_save, '.csv'), 'Delimiter',',');





end