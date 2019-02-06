from keras import backend as K
from NetworkHandler.KerasSupportMethods import AssertIsTensor, AssertTensorDotDim, AssertAddTensorToTensor

class ControledTensorOperations():
    def ControledConcatenation(prev_features, new_features, axis=1):
        AssertIsTensor(prev_features)
        AssertIsTensor(new_features)
        return K.concatenate([prev_features, new_features], axis)

    def ControledWeightDotProduct(aggregated_vectors, weights):
        AssertIsTensor(aggregated_vectors)
        AssertIsTensor(weights)
        AssertTensorDotDim(aggregated_vectors, weights)
        return K.dot(aggregated_vectors, weights)

    def ControledBiased(weighted, bias):
        AssertIsTensor(weighted)
        AssertIsTensor(bias)
        AssertAddTensorToTensor(weighted, bias) 
        return weighted + bias