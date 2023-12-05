%%
clc,clear,close all;
rng('shuffle');

% 初始化Bob和Eve的snr
Bob_train_snr = 10;
Eve_train_snr = 3;

% 初始化模型参数
innerLen = 24;
M=16;
modLen = 4;
outerLen = innerLen*2/modLen;
nEpochs = 1500;
BatchSize = 120;

% 创建学习速率调度器
initialLearnRate = 0.1; % 初始学习率
decayRate = 0.5; % 学习率衰减率
dropPeriod = 250; % 学习率衰减周期

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
SampleSize = 102400;
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
    'ExecutionEnvironment','auto',...
    'OutputFcn', outputFunction);

% 训练
Bob_autoencoder=trainNetwork(SampleData,SampleData,Bob_autoencoderLayers,options);
save('Bob_autoencoder.mat','Bob_autoencoder');
Eve_autoencoder=trainNetwork(SampleData,SampleData,Eve_autoencoderLayers,options);
save('Eve_autoencoder.mat','Eve_autoencoder');


%%
%TODO: KMEANS
%随机划分一个kmeans.cluster 等大小？
kmeans_labels = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4];
kmeans_labels = kmeans_labels(randperm(n));

gen_matrix=zeros(M,M);
for j=1:modLen
	for i=1:M
		if kmeans_labels(i)==j
			for k=1:M
				if kmeans_labels(k)==j
					gen_matrix(i,k)=1/modLen;
				end
			end
		end
	end
end

%第二轮物理层安全训练，只训练Bob，损失函数为Bob和Eve的损失加权
% 生成训练集
SampleSize2 = SampleSize;
SampleData2 = randi([0,1],innerLen,SampleSize2);
SampleData2_Eve=zeros(innerLen,SampleSize2);
for i=1:SampleSize2
	for j=1:innerLen/modLen
		label=bit2int(SampleData2(modLen*(j-1)+1:modLen*j,i),modLen);
		TempData_Eve=zeros(modLen,SampleSize2);
		for k=1:M
			if gen_matrix(label,k)>0
				Data_p=int2bit(k);
				TempData_Eve=TempData_Eve+gen_matrix(label,k)*Data_p;
			end
		end
		SampleData2_Eve(modLen*(j-1)+1:modLen*j,i)=TempData_Eve;
	end
end

% 自定义加权损失函数
Bob_Secure_alpha=0.7;
Eve_Secure_alpha=1-Bob_Secure_alpha;
Bob_predict2=Bob_autoencoder.predict(SampleData2);
Bob_train_mse=immse(Bob_predict2,SampleData2)/SampleSize2;
Eve_predict2=Eve_autoencoder.predict(SampleData2);
Eve_train_mse=immse(Eve_predict2,SampleData2_Eve)/SampleSize2;
Secure_mse=Bob_Secure_alpha*Bob_train_mse+Eve_Secure_alpha*Eve_train_mse;



% 设置训练选项
options2 = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', decayRate, ...
    'LearnRateDropPeriod', dropPeriod,...
    'MaxEpochs', nEpochs, ...
    'MiniBatchSize', BatchSize, ...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','auto',...
    'OutputFcn', outputFunction);

% 第二轮训练
gaussLayerIdx=8;
Bob_trainChannel=gaussianNoiseLayer(Bob_train_snr);
Bob_layerGraph=layerGraph(Bob_autoencoder);
Bob_layerGraph=replaceLayer(Bob_layerGraph,Bob_layerGraph.Layers(gaussLayerIdx).Name,Bob_trainChannel);
Bob_autoencoder=assembleNetwork(Bob_layerGraph);
Bob_autoencoder=trainNetwork(SampleData2,SampleData2,Bob_autoencoder,options2);
save('Bob_autoencoder2.mat','Bob_autoencoder');

%第二轮测试
load('Bob_autoencoder2.mat');
load('Eve_autoencoder.mat');
TestSize2=SampleSize2/10;
TestSnr2=-5:-1;
Bob_BER2 = zeros(1,length(TestSnr2));
Eve_BER2 = zeros(1,length(TestSnr2));
Bob_SER2 = zeros(1,length(TestSnr2));
Eve_SER2 = zeros(1,length(TestSnr2));
Bob_MSE2 = zeros(1,length(TestSnr2));
Eve_MSE2 = zeros(1,length(TestSnr2));

for i=1:length(TestSnr2)
    snr2=TestSnr(i);
    gaussLayerIdx2=8;
    Bob_testChannel2=gaussianNoiseLayer(snr2);
    Bob_layerGraph2=layerGraph(Bob_autoencoder);
    Bob_layerGraph2=replaceLayer(Bob_layerGraph2,Bob_layerGraph2.Layers(gaussLayerIdx2).Name,Bob_testChannel2);
    Bob_autoencoder2=assembleNetwork(Bob_layerGraph2);
    Eve_testChannel2=gaussianNoiseLayer(snr2-7);
    Eve_layerGraph2=layerGraph(Eve_autoencoder2);
    Eve_layerGraph2=replaceLayer(Eve_layerGraph2,Eve_layerGraph2.Layers(gaussLayerIdx2).Name,Eve_testChannel2);
    Eve_autoencoder2=assembleNetwork(Eve_layerGraph2);
    for j=1:TestSize2
        TestInfo2=randi([0,1],cfgLDPCEnc.NumInformationBits,1);
        TestData2=ldpcEncode(TestInfo2,cfgLDPCEnc);
        TestInput2=reshape(TestData2,innerLen,[]);
        Bob_output2=predict(Bob_autoencoder,TestInput2);
        Eve_output2=predict(Eve_autoencoder,TestInput2);
        Bob_predict2=double(reshape(Bob_output2,cfgLDPCEnc.BlockLength,[]));
        Eve_predict2=double(reshape(Eve_output2,cfgLDPCEnc.BlockLength,[]));

        Bob_mse2=immse(Bob_predict2,TestData2)/TestSize2;
        Eve_mse2=immse(Eve_predict2,TestData2)/TestSize2;
        Bob_MSE2(i)=Bob_MSE2(i)+Bob_mse2;
        Eve_MSE2(i)=Eve_MSE2(i)+Eve_mse2;

        Bob_data2=double(Bob_predict2>0.5);
        Eve_data2=double(Eve_predict2>0.5);
        [~,ser12]=biterr(Bob_data2,TestData2);
        Bob_ser2=ser12/TestSize2;
        [~,ser22]=biterr(Eve_data2,TestData2);
        Eve_ser2=ser22/TestSize2;
        Bob_SER2(i)=Bob_SER2(i)+Bob_ser2;
        Eve_SER2(i)=Eve_SER2(i)+Eve_ser2;

        Bob_llr2=Bob_predict2;
        Eve_llr2=Eve_predict2;
        for k=1:length(Bob_llr2)
            if Bob_llr2(k)<1e-8
                Bob_llr2(k)=1e-8;
            elseif Bob_llr2(k)>1-1e-8
                Bob_llr2(k)=1-1e-8;
            end
            Bob_llr2(k)=log((1-Bob_llr2(k))/Bob_llr2(k));
        end
        for k=1:length(Eve_llr)
            if Eve_llr2(k)<1e-8
                Eve_llr2(k)=1e-8;
            elseif Eve_llr2(k)>1-1e-8
                Eve_llr2(k)=1-1e-8;
            end
            Eve_llr2(k)=log((1-Eve_llr2(k))/Eve_llr2(k));
        end
        Bob_Info2=ldpcDecode(Bob_llr2,cfgLDPCDec,maxnumiter);
        Eve_Info2=ldpcDecode(Eve_llr2,cfgLDPCDec,maxnumiter);
        [~,ber12]=biterr(Bob_Info2,TestInfo2);
        Bob_ber2=ber12/TestSize2;
        [~,ber22]=biterr(Eve_Info2,TestInfo2);
        Eve_ber2=ber22/TestSize2;
        Bob_BER2(i)=Bob_BER2(i)+Bob_ber2;
        Eve_BER2(i)=Eve_BER2(i)+Eve_ber2;
    end
end

figure;
semilogy(TestSnr2,Bob_MSE2,TestSnr2,Eve_MSE2);
title('MSE2');
xlabel('SNR');
ylabel('MSE2');
legend('Bob','Eve');
grid on
savefig('MSE2.fig');

figure;
semilogy(TestSnr2,Bob_SER2,TestSnr2,Eve_SER2);
title('SER2');
xlabel('SNR');
ylabel('SER2');
legend('Bob','Eve');
grid on
savefig('SER2.fig');

figure;
semilogy(TestSnr2,Bob_BER2,TestSnr2,Eve_BER2);
title('BER2');
xlabel('SNR');
ylabel('BER2');
legend('Bob','Eve');
grid on
savefig('BER2.fig');