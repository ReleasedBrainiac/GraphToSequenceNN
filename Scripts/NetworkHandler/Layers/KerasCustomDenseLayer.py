import keras
from keras import backend as K
from keras import activations, regularizers
from keras.layers import Layer

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/layers.py

    Extracted the implementation strategy of the IBM example for Keras API users.
    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.

    Basics for CustomLayer resource: https://keras.io/layers/writing-your-own-keras-layers/
    Further resources: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
'''

#TODO call function maybe false implemented

class KerasCustomDense(Layer):
    """
    This class implements a simple custom dense layer.
    Its abstracted from the IBM Tensorflow example.
    ATTENTION: The sparse input option and the featureless option were removed caused by currently missing necessity.
        :param Layer: keras layer definition
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 name='kernel_weights',
                 weight_init='glorot_uniform',
                 dropout=0.,
                 activation=activations.relu, 
                 bias=True, 
                 bias_init='zeros',
                 **kwargs):
        """
        This constructor collects all necessary information to build the layer in the  following steps.
            :param input_dim: dimension of the input 
            :param output_dim: dimension of the output
            :param name: name of the layer
            :param weight_init: init function to generate the layer weights
            :param dropout: dropout rate
            :param activation: activation function
            :param bias: boolean to use bias
            :param bias_init: function to generate bias weights
            :param **kwargs: additional args
        """
        super(KerasCustomDense, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.kernel_initializers = weight_init
        self.bias = bias
        self.bias_initializers = bias_init
        self.dropout = dropout
        self.activation = activation

    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        In this case the function generate the bias and layer kernel weights.
        ATTENTION: The weight decay is fixed with = 0.000, Later a dynamic call via config is maybe possible.
            :param input_shape: shape of the input tensor
        """
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name=self.name+'_weights',
                                      shape=(self.input_dim, self.output_dim),
                                      dtype='float32',
                                      initializer=self.kernel_initializers,
                                      regularizer=regularizers.l2(0.000),
                                      trainable=True)

        self.bias_weights = self.add_weight(name=self.name+'_bias',
                                            shape=self.output_dim,
                                            initializer=self.bias_initializers)

        super(KerasCustomDense, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """
        assert isinstance(input_shape, list)
        shape_feats, shape_edges = input_shape
        return (shape_feats[0], self.output_dim)
    
    def call(self, inputs):
        """
        This function process the layer logic.
        Here the dropout will calculated and the result will passed through activation after multiplication with the weights.
        ATTENTION: The dropout was commented out in the origin version so just keep this in mind.
            :param inputs: 
        """ 
        assert isinstance(inputs, list)
        inputs[0] = K.dropout(inputs[0], self.dropout)
        output = K.dot(inputs[0], self.kernel)
        if self.bias: output += self.bias_weights
        return self.activation(output)
