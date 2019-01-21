from keras import backend as K
from keras.layers import Layer
from keras import initializations, activations

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

# TODO Das muss in die Arbeit =>  Varianz https://github.com/keras-team/keras/issues/9779 

#TODO try catch and documentation
class KerasCustomMaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions."""
    def __init__(   self, 
                    input_dim, 
                    output_dim, 
                    model_size="small", 
                    neigh_input_dim=None,
                    dropout=0., 
                    bias=True, 
                    activation=activations.relu, 
                    name='', 
                    concat=False, **kwargs):

        super(KerasCustomMaxPoolingAggregator, self).__init__(**kwargs)

        self.hop_layers = []
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.concat = concat

        
        if name is not None: name = '/' + name 
        if neigh_input_dim is None: neigh_input_dim = input_dim
        if concat: self.output_dim = 2 * output_dim

        hidden_dim = 50 if model_size == "small" else 100
        self.hidden_dim = hidden_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

        
        self.hop_layers.append(Dense(   input_dim=neigh_input_dim, 
                                        output_dim=hidden_dim, 
                                        act=activation,
                                        dropout=dropout, 
                                        sparse_inputs=False, 
                                        logging=self.logging))        

    def build(self, input_shape):
        self.self_node_weights = self.add_weight(   name=self.name+'_self_node_weights', 
                                                    shape=(self.input_dim, self.output_dim),
                                                    initializer='glorot_uniform',
                                                    trainable=True)

        self.neigh_node_weights = self.add_weight(  name=self.name+'_neigh_node_weights', 
                                                    shape=(self.hidden_dim, self.output_dim),
                                                    initializer='glorot_uniform',
                                                    trainable=True)

        self.bias = self.add_weight(name=self.name+'_bias',
                                    shape=self.output_dim,
                                    initializer='zeros')
        super(KerasCustomMaxPoolingAggregator, self).build(input_shape)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def call(self, inputs):
        # [current_vecs, neigh_vecs]
        self_node_vecs, neigh_node_vecs = inputs

        batch_size = neigh_node_vecs.shape[0]
        num_neighbors = neigh_node_vecs.shape[1]
        hidden_reshaped = K.reshape(neigh_node_vecs, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.hop_layers: hidden_reshaped = l(hidden_reshaped)

        neigh_node_vecs = K.reshape(hidden_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_node_vecs = K.max(neigh_node_vecs, axis=1)

        from_neighs = K.dot(neigh_node_vecs, self.neigh_node_weights)
        from_self = K.dot(self_node_vecs, self.self_node_weights)

        output = (from_self + from_neighs) if (not self.concat) else K.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias: output += self.vars['bias']
        return self.act(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)