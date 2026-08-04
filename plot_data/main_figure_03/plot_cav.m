clear all
close all
clc

load cavity_data.mat
set(0, 'DefaultFigurePosition', [50, 50, 1200, 650]);

circuit_mod = [114.4314	99.9280	63.3633 58.5921];

%inWall_model = readmatrix("inwallCavity_data.txt");

figure(1)
hold all
plot(HBKMeasurement(:,4),HBKMeasurement(:,3),'color',[0 0.4470 0.7410],'LineWidth',2)
plot(HBKMeasurement(:,4),HBKMeasurement(:,2),'color',[0.8500 0.3250 0.0980],'LineWidth',2)
plot(HBKMeasurement(:,4),HBKMeasurement(:,1),'color',[0.9290 0.6940 0.1250],'LineWidth',2)

plot(ModelData(:,1),ModelData(:,2),'o','color',[0 0.4470 0.7410],'LineWidth',2)
plot(ModelData(:,1),ModelData(:,4),'o','color',[0.8500 0.3250 0.0980],'LineWidth',2)
plot(ModelData(:,1),ModelData(:,6),'o','color',[0.9290 0.6940 0.1250],'LineWidth',2)
%plot(inWall_model(:,1),inWall_model(:,2),'o','color',[1 0.38820 1],'LineWidth',2)

plot([circuit_mod(3) circuit_mod(3)],[0.2 12],'--','color',[0 0.4470 0.7410],'LineWidth',2)
plot([circuit_mod(2) circuit_mod(2)],[0.2 12],'--','color',[0.8500 0.3250 0.0980],'LineWidth',2)
plot([circuit_mod(1) circuit_mod(1)],[0.2 12],'--','color',[0.9290 0.6940 0.1250],'LineWidth',2)
%plot([circuit_mod(4) circuit_mod(4)],[0.2 12],'--','color',[1 0.38820 1],'LineWidth',2)

grid on
xlabel 'Frequency [Hz]'
ylabel 'Gain [P_c_a_v / P_r_e_f]'
%title 'Simulated and Measured Resonance of Cavity'
legend('Measured Large','Measured Medium','Measured Small','Simulated Large','Simulated Medium','Simulated Small','Circuit Large','Circuit Medium','Circuit Small')
fontsize(17,"points")
ylim([0,8])