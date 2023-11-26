clc,clear,close all;
rng('shuffle');

P=[0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1
    6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1
    2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1
    23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1
    24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
    25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
    13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1
    7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1
    11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1
    25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0
    3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];
blockSize = 27;

%配置LDPC编译码
pcmatrix = ldpcQuasiCyclicMatrix(blockSize,P);
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix);

%BP最大迭代次数
maxnumiter=100;

% 初始化Bob和Eve的snr
Bob_train_snr = 10;
Eve_train_snr = 3;

% 初始化模型参数
innerLen = 24;
modLen = 4;
outerLen = innerLen*2/modLen;
nEpochs = 50;
BatchSize = 120;

% 创建学习速率调度器
initialLearnRate = 0.01; % 初始学习率
decayRate = 0.5; % 学习率衰减率
dropPeriod = 50; % 学习率衰减周期

% 创建输出函数
outputFunction = @(info) disp(['Epoch: ' num2str(info.Epoch) ', Loss: ' num2str(info.TrainingLoss)]);

% 构建编码器部分
Alice_encoder = [ ...
    sequenceInputLayer(innerLen,"SplitComplexInputs",true)
    fullyConnectedLayer(innerLen*2)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(outerLen)
    reluLayer
    batchNormalizationLayer];

% 构建噪声层
Bob_channel = gaussianNoiseLayer(Bob_train_snr);
Eve_channel = gaussianNoiseLayer(Eve_train_snr);

% 构建译码器部分
Bob_decoder = [ ...
    fullyConnectedLayer(innerLen*2)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(innerLen)
    reluLayer
    batchNormalizationLayer
    sigmoidLayer
    regressionLayer];

Eve_decoder = [ ...
    fullyConnectedLayer(innerLen*2)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(innerLen)
    reluLayer
    batchNormalizationLayer
    sigmoidLayer
    regressionLayer];

% 构建自动编码器网络
Bob_autoencoderLayers = [ ...
    Alice_encoder
    Bob_channel
    Bob_decoder];
Eve_autoencoderLayers = [...
    Alice_encoder
    Eve_channel
    Eve_decoder];

% 第一轮训练
% 生成训练集
SampleSize = 1024;
SampleData = randi([0,1],innerLen,SampleSize);

% 设置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', decayRate, ...
    'LearnRateDropPeriod', dropPeriod,...
    'MaxEpochs', nEpochs, ...
    'MiniBatchSize', BatchSize, ...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','gpu',...
    'OutputFcn', outputFunction);

% 训练
Bob_autoencoder=trainNetwork(SampleData,SampleData,Bob_autoencoderLayers,options);
Eve_autoencoder=trainNetwork(SampleData,SampleData,Eve_autoencoderLayers,options);

%测试
TestSize=10;
TestSnr=0:5:15;
Bob_BER = zeros(1,length(TestSnr));
Eve_BER = zeros(1,length(TestSnr));
Bob_SER = zeros(1,length(TestSnr));
Eve_SER = zeros(1,length(TestSnr));
Bob_MSE = zeros(1,length(TestSnr));
Eve_MSE = zeros(1,length(TestSnr));

for i=1:length(TestSnr)
    snr=TestSnr(i);
    gaussLayerIdx=8;
    Bob_testChannel=gaussianNoiseLayer(snr);
    Bob_layerGraph=layerGraph(Bob_autoencoder);
    Bob_layerGraph=replaceLayer(Bob_layerGraph,Bob_layerGraph.Layers(gaussLayerIdx).Name,Bob_testChannel);
    Bob_autoencoder=assembleNetwork(Bob_layerGraph);
    Eve_testChannel=gaussianNoiseLayer(snr-7);
    Eve_layerGraph=layerGraph(Eve_autoencoder);
    Eve_layerGraph=replaceLayer(Eve_layerGraph,Eve_layerGraph.Layers(gaussLayerIdx).Name,Eve_testChannel);
    Eve_autoencoder=assembleNetwork(Eve_layerGraph);
    for j=1:TestSize
        TestInfo=randi([0,1],cfgLDPCEnc.NumInformationBits,1);
        TestData=ldpcEncode(TestInfo,cfgLDPCEnc);
        TestInput=reshape(TestData,innerLen,[]);
        Bob_output=predict(Bob_autoencoder,TestInput);
        Eve_output=predict(Eve_autoencoder,TestInput);
        Bob_predict=double(reshape(Bob_output,cfgLDPCEnc.BlockLength,[]));
        Eve_predict=double(reshape(Eve_output,cfgLDPCEnc.BlockLength,[]));

        Bob_mse=immse(Bob_predict,TestData)/TestSize/cfgLDPCEnc.BlockLength;
        Eve_mse=immse(Eve_predict,TestData)/TestSize/cfgLDPCEnc.BlockLength;
        Bob_MSE(i)=Bob_MSE(i)+Bob_mse;
        Eve_MSE(i)=Eve_MSE(i)+Eve_mse;

        Bob_data=double(Bob_predict>0.5);
        Eve_data=double(Eve_predict>0.5);
        [~,ser1]=biterr(Bob_data,TestData);
        Bob_ser=ser1/TestSize;
        [~,ser2]=biterr(Eve_data,TestData);
        Eve_ser=ser2/TestSize;
        Bob_SER(i)=Bob_SER(i)+Bob_ser;
        Eve_SER(i)=Eve_SER(i)+Eve_ser;

        Bob_llr=Bob_predict;
        Eve_llr=Eve_predict;
        for k=1:length(Bob_llr)
            if Bob_llr(k)<1e-8
                Bob_llr(k)=1e-8;
            elseif Bob_llr(k)>1-1e-8
                Bob_llr(k)=1-1e-8;
            end
            Bob_llr(k)=log((1-Bob_llr(k))/Bob_llr(k));
        end
        for k=1:length(Eve_llr)
            if Eve_llr(k)<1e-8
                Eve_llr(k)=1e-8;
            elseif Eve_llr(k)>1-1e-8
                Eve_llr(k)=1-1e-8;
            end
            Eve_llr(k)=log((1-Eve_llr(k))/Eve_llr(k));
        end
        Bob_Info=ldpcDecode(Bob_llr,cfgLDPCDec,maxnumiter);
        Eve_Info=ldpcDecode(Eve_llr,cfgLDPCDec,maxnumiter);
        [~,ber1]=biterr(Bob_Info,TestInfo);
        Bob_ber=ber1/TestSize;
        [~,ber2]=biterr(Eve_Info,TestInfo);
        Eve_ber=ber2/TestSize;
        Bob_BER(i)=Bob_BER(i)+Bob_ber;
        Eve_BER(i)=Eve_BER(i)+Eve_ber;
    end
end

figure;
semilogy(TestSnr,Bob_MSE,TestSnr,Eve_MSE);
title('MSE');
xlabel('SNR');
ylabel('MSE');
legend('Bob','Eve');
grid on

figure;
semilogy(TestSnr,Bob_SER,TestSnr,Eve_SER);
title('SER');
xlabel('SNR');
ylabel('SER');
legend('Bob','Eve');
grid on

figure;
semilogy(TestSnr,Bob_BER,TestSnr,Eve_BER);
title('BER');
xlabel('SNR');
ylabel('BER');
legend('Bob','Eve');
grid on