clear
close all
clc

load HT06filt.mat

NFFT = length(y_filt);
spec = abs(fft(y_filt, NFFT));
Fs = 2000;
f = Fs*linspace(0, 1, NFFT);
semilogy(f, spec./length(y_filt))
xlim([0 65])
grid on
xlabel 'Frequency [Hz]'
ylabel 'Normalized FFT [arb]'