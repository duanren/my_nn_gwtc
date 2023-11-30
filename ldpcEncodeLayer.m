% 定义LDPC编码层的自定义类
classdef ldpcEncodeLayer < nnet.layer.Layer
    properties
        H
		cfgLDPCEnc
    end

    methods
        function layer = ldpcEncodeLayer(h)
            layer.H = h;
			layer.cfgLDPCEnc = ldpcEncoderConfig(h);
        end

        function Z = predict(layer, X)
            Z = ldpcEncode(X, layer.cfgLDPCEnc);
        end
    end
end
