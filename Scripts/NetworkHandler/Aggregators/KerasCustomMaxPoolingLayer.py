from keras import backend as K
from keras.layers import Layer
from keras import initializations, activations
from NetworkHandler.Layers import KerasCustomDenseLayer as Dense

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

#TODO Das muss in die Arbeit =>  Varianz https://github.com/keras-team/keras/issues/9779
#TODO ATTENTION:  The order of [3] and [4] is the other way around provide in the paper.
#TODO documentation

class KerasCustomMaxPoolingAggregator(Layer):
    """ This class aggregates via max-pooling and concatenation over a graph neighbourhood."""

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
        """
        This constructor initializes all necessary variables. Except input_dim and output_dim, all parameters have preset values.
            :param input_dim: 
            :param output_dim: 
            :param model_size: 
            :param neigh_input_dim: 
            :param dropout: 
            :param bias: 
            :param activation: 
            :param name: 
            :param concat: 
            :param **kwargs: 
        """

        super(KerasCustomMaxPoolingAggregator, self).__init__(**kwargs)

        self.layers = []
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.activation = activation
        self.concat = concat
        self.input_dim = input_dim


        """ Care these are ternary if cases """
        self.name = '/' + name if (name is not None) else ''  
        self.neigh_input_dim = input_dim if (neigh_input_dim is None) else neigh_input_dim
        self.output_dim = 2 * output_dim if (concat) else output_dim
        self.hidden_dim = 50 if model_size == "small" else 100

        self.layers.append(Dense(   input_dim=self.neigh_input_dim, 
                                    output_dim=self.hidden_dim, 
                                    activation=self.activation,
                                    dropout=self.dropout, 
                                    sparse_inputs=False))        



    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        This includes the matrices for the current and neighbourhood node(s) and also the bias weight matrix. 
            :param input_shape: shape of the input tensor
        """
        self.self_node_weights = self.add_weight(   name=self.name+'_self_node_weights', 
                                                    shape=(self.input_dim, self.output_dim),
                                                    initializer='glorot_uniform',
                                                    trainable=True)

        self.neigh_node_weights = self.add_weight(  name=self.name+'_neigh_node_weights', 
                                                    shape=(self.hidden_dim, self.output_dim),
                                                    initializer='glorot_uniform',
                                                    trainable=True)

        self.bias_weights = self.add_weight(name=self.name+'_bias',
                                            shape=self.output_dim,
                                            initializer='zeros')
        super(KerasCustomMaxPoolingAggregator, self).build(input_shape)

    def call(self, inputs):
        """
        This function keeps the KerasCustomMaxPoolingAggregator layer logic.
        [1] The layer reshapes the input depending on the input and batch size.
        [2] Calculate max pooling
        [3] Caclulate weight matrix multiplication
        [4] Calculate concatenation
        [5] Optional: Add bias
        [6] Process Activation
        
            :param inputs: layer input tensors
        """   
        self_node_vecs, neigh_node_vecs = inputs
        batch_size = neigh_node_vecs.shape[0]
        num_neighbors = neigh_node_vecs.shape[1]

        """ [1] """
        hidden_reshaped = K.reshape(neigh_node_vecs, (batch_size * num_neighbors, self.neigh_input_dim))
        for l in self.layers: hidden_reshaped = l(hidden_reshaped)
        
        """ [2] """
        neigh_node_vecs = K.reshape(hidden_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_node_vecs = K.max(neigh_node_vecs, axis=1)
        
        """ [3] """
        from_neighs = K.dot(neigh_node_vecs, self.neigh_node_weights)
        from_self = K.dot(self_node_vecs, self.self_node_weights)

        """ [4] """
        output = (from_self + from_neighs) if (not self.concat) else K.concatenate([from_self, from_neighs], axis=1)

        """ [5] """
        if self.bias: output += self.bias_weights
        
        """ [6] """
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """   
        return (input_shape[0], self.output_dim)