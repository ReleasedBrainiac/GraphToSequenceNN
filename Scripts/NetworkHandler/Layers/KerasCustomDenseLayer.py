import tensorflow as tf
import keras
from keras import backend as K
from keras import initializations, activations
from keras.layers import Layer

from inits import zeros
import configure as conf
#from BasicLayer import Layer

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/layers.py

    Extracted the implementation strategy of the IBM example for Keras API users.
    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.

    Basics for CustomLayer resource: https://keras.io/layers/writing-your-own-keras-layers/
'''


#TODO try catch and documentation
class KerasCustomDense(Layer):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 name='kernel_weights',
                 weight_init='glorot_uniform',
                 dropout=0.,
                 activation=activations.relu, 
                 placeholders=None, 
                 bias=True, 
                 bias_init='zeros',
                 featureless=False,
                 sparse_inputs=False, 
                 **kwargs):
        """
        docstring here
            :param self: 
            :param input_dim: 
            :param output_dim: 
            :param name='kernel_weights': 
            :param weight_init='glorot_uniform': 
            :param dropout=0.: 
            :param activation=activations.relu: 
            :param placeholders=None: 
            :param bias=True: 
            :param bias_init='zeros': 
            :param featureless=False: 
            :param sparse_inputs=False: 
            :param **kwargs: 
        """


        super(KerasCustomDense, self).__init__(**kwargs)

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.kernel_initializers = initializations.get(weight_init)
        self.bias = bias
        self.bias_initializers = initializations.get(bias_init)
        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        
        

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs: self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable( 'weights', 
                                                    shape=(input_dim, output_dim),
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    regularizer=tf.contrib.layers.l2_regularizer(conf.weight_decay))

            if self.bias: self.vars['bias'] = zeros([output_dim], name='bias')


    # Here we build the weight matrix
    def build(self, input_shape):
        self.kernel = self.add_weight(name=self.name, 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer=keras.initializers.glorot_normal(seed=None),
                                      trainable=True)
        super(KerasCustomDense, self).build(input_shape)


    # Here lives the layer logic part
    def call(self, x):
        return K.dot(x, self.kernel)

    # Here lives the output shape transformation logic
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)







    def _call(self, inputs):
        x = inputs

        # x = tf.nn.dropout(x, self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias: output += self.vars['bias']

        return self.activation(output)
