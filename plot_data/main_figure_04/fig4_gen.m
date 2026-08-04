clear all
close all
clc

load HT06ex.mat
load HT06filt.mat

plot(0.00025*(0:15998),ECG_raw)
hold all
plot(0.00025*(0:15998),Audio_raw-3)
plot(0.00025*(0:2:16000-1),HRITest0610(:,2)-1,'Color',[0 0.8 0],'LineWidth',2)
plot(0.00025*(200:2:16000),y_filt/max(y_filt)-2)
xlabel('Seconds [s]')

figure(2)

NFFT = length(y_filt);
spec = abs(fft(y_filt,NFFT));
Fs = 2000;
f = Fs*linspace(0,1,NFFT);
semilogy(f,spec./length(y_filt))
xlim([0 65])
grid on
xlabel 'Frequency [Hz]'
ylabel 'Normalized FFT [arb]'

