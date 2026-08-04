% script to list all files in the directory, calculate their sizes to print
% the total dataset length 

clear all
close all
clc 

dir_root = 'C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\all data raw\';

list_txt_files = dir(fullfile(dir_root, '*.txt'));

fs = 4e3;

line_count = zeros(size(list_txt_files,1),1);

disp('DURATION OF INDIVIDUAL RECORDINGS: ')
for i=1:size(list_txt_files,1)
    file_name = list_txt_files(i).name;

    fid = fopen(strcat(dir_root, file_name), 'r');


    while ~feof(fid)
        fgets(fid);  % Read one line
        line_count(i) = line_count(i) + 1;
    end
    
    %removing 9 rows to eliminate overhead
    %dividing by fs to get value in seconds
    line_count(i) = (line_count(i)-9)/fs;

    disp(file_name);
    disp(strcat(num2str(line_count(i)), ' sec'))

end

%getting total value in minutes
dataset_duration_min = sum(line_count)/60;

disp(strcat('total dataset duration: ', num2str(dataset_duration_min), ' minutes'));