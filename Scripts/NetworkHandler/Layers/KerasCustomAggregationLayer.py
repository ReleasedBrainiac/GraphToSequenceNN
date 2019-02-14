from keras import backend as K
from NetworkHandler.KerasSupportMethods.SupportMethods import KerasEval as KE
from NetworkHandler.KerasSupportMethods.ControledBasicOperations import ControledTensorOperations as CTO
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from keras import activations
from keras.layers import Layer

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    Attention: 
        1. The mean calc is implemented like the Keras GlobalAveragePooling1D (without masked sum masking)!
        2. The max pooling is implemented like the Keras GlobalMaxPooling1D 
        => https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557
        => https://keras.io/layers/pooling/
'''

#TODO IN MA => Varianz https://github.com/keras-team/keras/issues/9779
#TODO IN MA => Exakte paper implementation wie im paper im gegensatz zum beispiel code

class CustomAggregationLayer(Layer):

    """ This class aggregates via mean and max calculation and also concatenation over a graph neighbourhood."""

    def __init__(   self, 
                    input_dim, 
                    output_dim,
                    activation=activations.relu, 
                    aggregator='mean',
                    **kwargs):
        """
        This constructor initializes all necessary variables. Except input_dim and output_dim, all parameters have preset values.
            :param input_dim: describes the maximal possible size of a input
            :param output_dim: desired amount of neurons
            :param activation: neuron activation function (default = relu)
            :param aggregator: neighbourhood aggregation function (default = mean)
        """

        super(CustomAggregationLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.aggregator = aggregator
        
        

    def build(self, input_shape):
        """
        This function provides all necessary weight matrices for the layer.
        This includes the matrices for the current and neighbourhood node(s) and also the bias weight matrix. 
            :param input_shape: list of shapes of the input tensors
        """
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(  name='kernel',
                                        shape=(self.input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)

        super(CustomAggregationLayer, self).build(input_shape)

    def call(self, inputs):
        """
        This function keeps the CustomAggregationLayer layer logic.
        [0] Drop out on train state
        [1] Calculate aggregation
        [2] Calculate concatenation
        [3] Add zeros if the gradient has higher dimension
        [4] Caclulate weight matrix multiplication
        [5] Process Activation
            :param inputs: layer input tensors
        """   

        assert isinstance(inputs, list)
        features, edge_look_up = inputs

        """ [1] """
        agg_neigh_vecs = Nhood(features, edge_look_up, aggregator=self.aggregator).Execute()

        """ [2] """
        output = CTO.ControledConcatenation(features, agg_neigh_vecs)

        """ [3] """
        difference = CTO.ControledShapeDifference(self.kernel, output)
        assert (difference == 0) , ("Concatenation can't be weighted since it's shape is incompatible for K.dot operation => Shapes [Kernel",self.kernel.shape," | Concatenation",output)

        """ [4] """
        output = CTO.ControledWeightDotProduct(output, self.kernel)

        """ [5] """
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
        This function provides the layers output shape transformation logic.
            :param input_shape: tensor input shape
        """   
        assert isinstance(input_shape, list)
        shape_feats, shape_edges = input_shape
        return [(shape_feats[0], self.output_dim), shape_edges]
