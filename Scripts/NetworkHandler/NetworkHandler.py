# - *- coding: utf-8*-
'''
    Used Resources:
        => https://www.tensorflow.org/api_docs/python/tf
        => https://keras.io/getting-started/functional-api-guide/
'''

from LambdaNodeEmbedding import GetFeaturesTensorOnly
from keras.layers import Input, Dense, Lambda

'''
    This class library provide a pipeline to use keras, to build the model pipeline for Graph2Sequence encoding decoding.
'''

def GetKerasInputLayerPairs(shape_in1, name1, shape_in2, name2):
    inputs1 = Input(shape=shape_in1, name=name1)
    inputs2 = Input(shape=shape_in2, name=name2)
    return [inputs1, inputs2]

def GetMyKerasLambda(tensor_pair_array, out_shape, layer_name):
    return Lambda(GetFeaturesTensorOnly, output_shape=out_shape, name=layer_name)(tensor_pair_array)

def GetTestDense(units, activation_str, name, prev_layer):
    return Dense(units, activation=activation_str, name=name)(prev_layer)
