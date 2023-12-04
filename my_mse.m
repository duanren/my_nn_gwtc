function [loss,gradient,score] = my_mse(autoencoder,TrainData,TargetData)
    PredictData=forward(autoencoder,TrainData);

    loss=immse(PredictData,TargetData);

    gradient=dlgradient(loss,autoencoder.Learnables);

    score=exp(-loss);
end

