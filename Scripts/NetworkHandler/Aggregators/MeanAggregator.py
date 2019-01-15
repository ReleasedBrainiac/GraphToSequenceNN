import tensorflow as tf
from layers import Layer, Dense
from inits import glorot, zeros
from pooling import mean_pool

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

#TODO try catch and documentation
class MeanAggregator(Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, neigh_len = inputs

        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)