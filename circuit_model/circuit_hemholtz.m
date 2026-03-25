%%%%%%%%%%%%
% circuit_hemholtz.m
% Calculate the resonant frequency of multi-neck hemholtz resonator using
% circuit model
% P Schmalenberg
% MIT License
%%%%%%%%%%%

clear all
close all
clc

c = 340290; %[mm/s]

r_cyl = [50 50 70 50]; %[mm] Radius of cavity
h_cyl = [60 70 80 60]; %[mm] Height of cavity
V_cyl = pi*r_cyl.*r_cyl.*h_cyl; %Volume of cavity
neck_num = [12 12 12 8]; %number of necks in resonator

r_neck = [1.5 1.5 1.5 2.5]; %[mm] Radius of neck
A_neck = pi*r_neck.*r_neck.*neck_num; %Calculate total area of necks into resonator

L_neck = [36 41 46 280]; %[mm] Length of Neck
L_neck = L_neck + 0.24.*r_neck.*neck_num; %apply end correction based on [49]

S = c.^2 * A_neck.^2 ./ V_cyl;
m = A_neck.* L_neck;

f = (1/(2*pi)).*sqrt(S./m); %Circuit Model Resonant Frequency


