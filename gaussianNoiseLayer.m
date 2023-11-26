% 定义高斯噪声层的自定义类
classdef gaussianNoiseLayer < nnet.layer.Layer
    properties
        SNR % SNR
    end

    methods
        function layer = gaussianNoiseLayer(snr)
            layer.SNR = snr;
        end

        function Z = predict(layer, X)
            Z = awgn(X,layer.SNR);
        end
    end
end
