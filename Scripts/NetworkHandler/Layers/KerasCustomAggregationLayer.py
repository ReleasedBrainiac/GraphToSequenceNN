from keras import backend as K
from NetworkHandler.KerasSupportMethods import KerasEval as KE
from NetworkHandler.Aggregators.NeighbourAggregator import NeighbourAggregator as NAGG
from keras import activations
from keras.layers import Layer
from NetworkHandler.Layers import KerasCustomDenseLayer as Dense

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

#TODO Das muss in die Arbeit =>  Varianz https://github.com/keras-team/keras/issues/9779
#TODO ATTENTION:  The order of [3] and [4] is the other way around provide in the paper.
#TODO missing documentation

class CustomAggregationLayerSimple(Layer):

    """ This class aggregates via mean and max calculation and also concatenation over a graph neighbourhood."""

    def __init__(   self, 
                    input_dim, 
                    output_dim, 
                    neigh_input_dim=None,
                    dropout=0., 
                    bias=True, 
                    activation=activations.relu, 
                    name=None, 
                    concat=False, 
                    mode='train',
                    aggregator='mean',
                    **kwargs):
        """
        This constructor initializes all necessary variables. Except input_dim and output_dim, all parameters have preset values.
            :param input_dim: 
            :param output_dim: 
            :param neigh_input_dim: 
            :param dropout: 
            :param bias: 
            :param activation: 
            :param name: 
            :param concat:
            :param mode: 
            :param aggregator: 
            :param **kwargs: 
        """

        super(CustomAggregationLayerSimple, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.concat = concat
        self.input_dim = input_dim
        self.mode = mode
        self.aggregator = aggregator


        """ Care these are ternary if cases """
        self.name = '/' + name if (name is not None) else ''  
        self.neigh_input_dim = input_dim if (neigh_input_dim is None) else neigh_input_dim
        self.output_dim = 2 * output_dim if (concat) else output_dim

    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        This includes the matrices for the current and neighbourhood node(s) and also the bias weight matrix. 
            :param input_shape: shape of the input tensor
        """
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(  name=self.name+'_weights', 
                                        shape=(self.input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)


        self.bias_weights = self.add_weight(name=self.name+'_bias',
                                            shape=self.output_dim,
                                            initializer='zeros')

        super(CustomAggregationLayerSimple, self).build(input_shape)

    def call(self, inputs):
        """
        This function keeps the CustomAggregationLayer layer logic.
        [1] Calculate aggregation
        [2] Calculate concatenation
        [3] Caclulate weight matrix multiplication
        [4] Optional: Add bias
        [5] Process Activation
        
            :param inputs: layer input tensors
        """   

        assert isinstance(inputs, list)
        features, embedding_look_up = inputs

        if self.mode: embedding_look_up = K.dropout(embedding_look_up, 1-self.dropout)

        AGGREGATOR = NAGG(features, embedding_look_up, aggregator=self.aggregator)

        """ [1] """
        aggregated_features = AGGREGATOR.Execute()

        """ [2] """
        if self.concat:
            output = AGGREGATOR.ControledConcatenation()
        else:
            output = AGGREGATOR.ControledAdd()

        """ [3] """
        output = AGGREGATOR.ControledWeightMult(output, self.kernel)

        """ [4] """
        if self.bias: output += self.bias_weights
        
        """ [5] """
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """   
        assert isinstance(input_shape, list)
        shape_feats, shape_edges = input_shape
        return (shape_feats[0], self.output_dim)

class CustomAggregationLayerMaxPool(Layer):

    """ This class aggregates via max-pooling layer and also concatenation over a graph neighbourhood."""

    def __init__(   self, 
                    input_dim, 
                    output_dim, 
                    model_size='small',
                    neigh_input_dim=None,
                    dropout=0., 
                    bias=True, 
                    activation=activations.relu, 
                    name=None, 
                    concat=False,
                    **kwargs):
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
            :param mode: 
            :param **kwargs: 
        """

        super(CustomAggregationLayerMaxPool, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.concat = concat
        self.input_dim = input_dim


        """ Care these are ternary if cases """
        self.name = '/' + name if (name is not None) else ''  
        self.neigh_input_dim = input_dim if (neigh_input_dim is None) else neigh_input_dim
        self.output_dim = 2 * output_dim if (concat) else output_dim
        self.hidden_dim = 50 if (model_size == 'small') else 50

        self.mp_layers = []
        self.mp_layers.append(Dense(input_dim=neigh_input_dim, 
                                    output_dim=self.hidden_dim, 
                                    activation=activation,
                                    dropout=dropout))

    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        This includes the matrices for the current and neighbourhood node(s) and also the bias weight matrix. 
            :param input_shape: shape of the input tensor
        """
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(  name=self.name+'_weights', 
                                        shape=(self.input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)


        self.bias_weights = self.add_weight(name=self.name+'_bias',
                                            shape=self.output_dim,
                                            initializer='zeros')

        super(CustomAggregationLayerMaxPool, self).build(input_shape)

    def call(self, inputs):
        """
        This function keeps the CustomAggregationLayer layer logic.
        [1] Calculate aggregation with max pool layer
        [2] Calculate concatenation
        [3] Caclulate weight matrix multiplication
        [4] Optional: add bias
        [5] Process Activation
        
            :param inputs: layer input tensors
        """   

        assert isinstance(inputs, list)
        features, embedding_look_up = inputs
        dims = embedding_look_up.shape
        batch_size = dims[0]
        max_neighbours = dims[1]
        AGGREGATOR = NAGG(features, embedding_look_up, aggregator='max')

        """ [1] """
        neigh_h = self.PerformMLPLayersCall(batch_size, max_neighbours)
        aggregated_features = AGGREGATOR.MaxPoolAggregator(neigh_h, axis=1)
        AGGREGATOR.OverwriteNewFeatures(new_features=aggregated_features)

        """ [2] """
        if self.concat:
            output = AGGREGATOR.ControledConcatenation()
        else:
            output = AGGREGATOR.ControledAdd()

        """ [3] """
        output = AGGREGATOR.ControledWeightMult(output, self.kernel)

        """ [4] """
        if self.bias: output += self.bias_weights
        
        """ [5] """
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """   
        assert isinstance(input_shape, list)
        shape_feats, shape_edges = input_shape
        return (shape_feats[0], self.output_dim)

    def PerformMaxPoolLayersCall(self, batch_size, max_neighbours):
        h_reshaped = K.reshape(neigh_h, (batch_size * max_neighbours, self.neigh_input_dim))
        for l in self.mp_layers: h_reshaped = l(h_reshaped)

        neigh_h = K.reshape(h_reshaped, (batch_size, max_neighbours, self.hidden_dim))