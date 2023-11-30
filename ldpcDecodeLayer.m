% 定义LDPC译码层的自定义类
classdef ldpcDecodeLayer < nnet.layer.Layer
    properties
        H
		cfgLDPCDec
		maxnumiter
    end

    methods
        function layer = ldpcEncodeLayer(h,iter=100)
            layer.H = h;
			layer.cfgLDPCDec = ldpcDecoderConfig(h);
			layer.maxnumiter = iter;
        end

        function Z = predict(layer, X)
            Z = ldpcDecode(X, layer.cfgLDPCDec, layer.maxnumiter);
        end
    end
end
