# - *- coding: utf-8*-
'''
    Used Resources:
        => https://www.tensorflow.org/api_docs/python/tf
        => https://keras.io/getting-started/functional-api-guide/
'''

from NetworkHandler.LambdaNodeEmbedding import GetFeaturesTensorOnly
from keras.layers import Input, Lambda

'''
    This class library provide a pipeline to use keras, to build the model pipeline for Graph2Sequence encoding decoding.
'''
class CustomLayerDefinitions:

    def GraphInputPairLayers(self, edges_shape=(None,), features_shape=(100,), name_edges='edges', name_features='features'):
        """
        This funtion returns a input layer pair for 2 given shapes.
        They will be used to generate tensor inputs from given data samples of structure [edges_array, nodes_features_array] 
            :param edges_shape: shape of the edge arrays (can be variable)
            :param features_shape: shape of the features (default value is caused by glove vector of length 100)
            :param name_edges: name for the edge layer
            :param name_features: name for the features layer
        """   
        try:
            edges_input_layer = Input(shape=edges_shape, name=name_edges)
            features_input_layer = Input(shape=features_shape, name=name_features)
            return [edges_input_layer, features_input_layer]
        except Exception as ex:
            template = "An exception of type {0} occurred in [CustomLayerDefinitions.GraphInputPairLayers]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    def CustomNodeLambdaLayer(self, input_layers, out_shape=(100,), layer_name='NodeEmbeddingLambda'):
        """
        This function returns a custom lambda layer to compute graphs on verticies and edges matrices.
            :param input_layers: 2 input layers for edges and features
            :param out_shape: shape of the custom layers output (default value is caused by glove vector of length 100)
            :param layer_name: name of the custom lambda layer
        """  
        try:
            return Lambda(GetFeaturesTensorOnly, output_shape=out_shape, name=layer_name)(input_layers)
        except Exception as ex:
            template = "An exception of type {0} occurred in [CustomLayerDefinitions.CustomNodeLambdaLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)      
