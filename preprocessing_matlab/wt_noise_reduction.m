%this fct expects one row of data to perform WT transform
%based on WT_NoiseReduction_MIRISE.m

function [y_out] = wt_noise_reduction(data, fs)

Lx = size(data, 1);

%WT parameter
DJ		= 1.947; % WT parameter of filterbank
PAD		= 0;
DT		= 1/fs;
SO		= DT; 
Jfac	= 2; 
J1		= round(Jfac*(log2(Lx*DT/SO))/DJ); 
MOTHER	= 'MORLET'; 
PARAM	= 6; 
                                                            %(Y,DT, PAD, DJ, S0, J1, MOTHER, PARAM)
[X, filter_freq, scale, coi, DJ, paramout, k_val] = contwt(data,DT, PAD, DJ, SO, J1, MOTHER, PARAM);

y_out = invcwt(X, MOTHER, scale, paramout, k_val); % y_out : output matrix of noise reduction 

y_out = y_out';

end