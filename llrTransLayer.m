% 定义llr变换层的自定义类
classdef llrTransLayer < nnet.layer.Layer
    properties
        Delta
    end

    methods
        function layer = gaussianNoiseLayer(delta)
            layer.Delta = delta;
        end

        function Z = predict(layer, X)
			Z=X;
			for k=1:length(Z)
				if Z(k)<layer.Delta
					Z(k)=layer.Delta;
				elseif Z(k)>1-layer.Delta
					Z(k)=1-layer.Delta;
				end
				Z(k)=log((1-Z(k))/Z(k));
			end
        end
    end
end
