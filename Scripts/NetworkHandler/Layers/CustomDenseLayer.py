import tensorflow as tf
from inits import zeros
import configure as conf
from BasicLayer import Layer

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/layers.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(conf.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # x = tf.nn.dropout(x, self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
