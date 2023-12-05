clc,clear,close all;
rng('shuffle');

% 加载第一次训练的模型
load('Bob_autoencoder.mat');
load('Eve_autoencoder.mat');

% 初始化模型参数
innerLen = 24;
M=16;
modLen = 4;
outerLen = innerLen*2/modLen;
nEpochs = 1500;
BatchSize = 120;



