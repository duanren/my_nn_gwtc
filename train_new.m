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
squaredGradientDecayFactor = 0.999;

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
    sigmoidLayer];

Eve_decoder = [ ...
    fullyConnectedLayer(innerLen*2)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(innerLen)
    reluLayer
    batchNormalizationLayer
    sigmoidLayer];

% 构建自动编码器网络
Bob_autoencoder = dlnetwork([ ...
    Alice_encoder
    Bob_channel
    Bob_decoder]);
Eve_autoencoder = dlnetwork([...
    Alice_encoder
    Eve_channel
    Eve_decoder]);

% 第一轮训练
% 生成训练集
SampleSize = 102400;
SampleData = randi([0,1],innerLen,SampleSize);

numObservationsTrain = SampleSize;
numIterationsPerEpoch = floor(numObservationsTrain/BatchSize);
numIterations = nEpochs*numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics=["Bob_Score","Eve_Score"], ...
    Info=["Epoch","Iteration"], ...
    XLabel="Iteration");

groupSubPlot(monitor,Score=["Bob_Score","Eve_Score"]);

epoch = 0;
iteration = 0;

trailingAvgBob = [];
trailingAvgSqBob = [];
trailingAvgEve = [];
trailingAvgSqEve = [];

numValidation = 27;
ZValidation = randi([0,1],innerLen,numValidation);


ZValidation = dlarray(ZValidation,"CB");

if canUseGPU
    ZValidation = gpuArray(ZValidation);
end

validationFrequency = 100;

% Loop over epochs.
while epoch < nEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Reset and shuffle datastore.
    SampleData=SampleData(:,randperm(SampleSize));

    batch = 0;

    % Loop over mini-batches.
    while batch*BatchSize<SampleSize && ~monitor.Stop
        iteration = iteration + 1;

        batch = batch + 1;
        % Read mini-batch of data.
        X = SampleData(:,(batch-1)*BatchSize+1:batch*BatchSize);

        X = dlarray(X,"CB");

        if canUseGPU
            X = gpuArray(X);
        end

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [Bob_loss,Bob_gradient,Bob_score] = dlfeval(@my_mse,Bob_autoencoder,X,X);
        [Eve_loss,Eve_gradient,Eve_score] = dlfeval(@my_mse,Eve_autoencoder,X,X);

        % Update the generator network parameters.
        [Bob_autoencoder,trailingAvgBob,trailingAvgSqBob] = adamupdate(Bob_autoencoder, Bob_gradient, ...
            trailingAvgBob, trailingAvgSqBob, iteration, ...
            initialLearnRate, decayRate, squaredGradientDecayFactor);

        % Update the discriminator network parameters.
        [Eve_autoencoder,trailingAvgEve,trailingAvgSqEve] = adamupdate(Eve_autoencoder, gradientsD, ...
            trailingAvgEve, trailingAvgSqEve, iteration, ...
            initialLearnRate, decayRate, squaredGradientDecayFactor);

        % Every validationFrequency iterations, display batch of generated
        % images using the held-out generator input.
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            % Generate images using the held-out generator input.
            XGeneratedValidation = predict(Bob_autoencoder,ZValidation);

        end

        % Update the training progress monitor.
        recordMetrics(monitor,iteration, ...
            Bob_Score=Bob_score, ...
            Eve_Score=Eve_score);

        updateInfo(monitor,Epoch=epoch,Iteration=iteration);
        monitor.Progress = 100*iteration/numIterations;
    end
end