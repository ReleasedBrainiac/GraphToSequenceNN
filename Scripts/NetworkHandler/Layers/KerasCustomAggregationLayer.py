from keras import backend as K
from NetworkHandler.KerasSupportMethods import KerasEval as KE
from NetworkHandler.KerasSupportMethods.ControledBasicOperations import ControledTensorOperations as CTO
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from keras import activations
from keras.layers import Layer, Dense, Dropout, GlobalMaxPooling1D

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.
'''

#TODO IN MA => Varianz https://github.com/keras-team/keras/issues/9779
#TODO IN MA => Exakte paper implementation wie im paper im gegensatz zum beispiel code
#TODO missing documentation and reference

class CustomAggregationLayerSimple(Layer):

    """ This class aggregates via mean and max calculation and also concatenation over a graph neighbourhood."""

    def __init__(   self, 
                    input_dim, 
                    output_dim, 
                    dropout=0., 
                    bias=True, 
                    activation=activations.relu, 
                    name=None, 
                    mode='train',
                    aggregator='mean',
                    **kwargs):
        """
        This constructor initializes all necessary variables. Except input_dim and output_dim, all parameters have preset values.
            :param input_dim: describes the maximal possible size of a input
            :param output_dim: desired amount of neurons
            :param dropout: ratio of values which should be randomly set to zero to prevent overfitting (default = 0.)
            :param bias: switch allow to use bias on the weighted result befor activation (default = True)
            :param activation: neuron activation function (default = relu)
            :param name: name of the layer (default = None)
            :param mode: mode the layer is running on (default = train)
            :param aggregator: neighbourhood aggregation function (default = mean)
        """

        super(CustomAggregationLayerSimple, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.name = '/' + name if (name is not None) else ''  
        self.mode = mode
        self.aggregator = aggregator
        
        

    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        This includes the matrices for the current and neighbourhood node(s) and also the bias weight matrix. 
            :param input_shape: list of shapes of the input tensors
        """
        assert isinstance(input_shape, list)
        feats_shape, neighs_shape = input_shape
        self.kernel = self.add_weight(  name=self.name+'_weights', 
                                        shape=(self.input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)


        self.bias_weights = self.add_weight(name=self.name+'_bias',
                                            shape=self.output_dim,
                                            initializer='zeros')

        self.zero_extender = self.add_weight(name=self.name+'_zeros',
                                             shape=input_shape[0][0],
                                             initializer='zeros')

        super(CustomAggregationLayerSimple, self).build(input_shape)

    def call(self, inputs):
        """
        This function keeps the CustomAggregationLayer layer logic.
        [0] Drop out on train state
        [1] Calculate aggregation
        [2] Calculate concatenation
        [3] Add zeros if the gradient has higher dimension
        [4] Caclulate weight matrix multiplication
        [5] Optional: Add bias
        [6] Process Activation
            :param inputs: layer input tensors
        """   

        assert isinstance(inputs, list)
        features, embedding_look_up = inputs

        if self.mode == 'train': 
            embedding_look_up = K.dropout(embedding_look_up, 1-self.dropout)

        """ [1] """
        agg_neigh_vecs = Nhood(features, embedding_look_up, aggregator=self.aggregator).Execute()

        """ [2] """
        output = CTO.ControledConcatenation(features, agg_neigh_vecs)

        """ [3] """
        difference = CTO.ControledShapeDifference(self.kernel, output)

        if difference > 0:
            output = CTO.ControlledMatrixExtension(output, self.zero_extender, difference)

        """ [4] """
        output = CTO.ControledWeightDotProduct(output, self.kernel)

        """ [5] """
        if self.bias:
            output = CTO.ControledBiased(output, self.bias_weights)

        """ [6] """
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """   
        assert isinstance(input_shape, list)
        shape_feats, shape_edges = input_shape
        return [(shape_feats[0], self.output_dim), shape_edges]
