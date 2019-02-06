from NetworkHandler.KerasSupportMethods.SupportMethods import AssertTensorShapeEqual, AssertIsTensor, AssertNotNone, AssertNotNegative
from NetworkHandler.AggregatorHandler.Aggregators import Aggregators
from keras import backend as K
from keras.layers import multiply

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    The original implementation snippeds from the IBM research team implemented in tensorflow can be found at:
        1. https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py

    The Keras resource can found here:
        1. https://keras.io/layers/merge/#multiply_1
        2. https://keras.io/backend/#backend
'''

class Neighbourhood():
    """
    This class calculates the neighborhood feature aggregation.
    """
    def __init__(self, features, neighbouring, axis=1, aggregator='mean', mp_layer=None, batch_size=None, neight_dim=None, hidden_dim=None):
        """
        This constructor collect all necessary variables for the setup.
        If [max_pool] is the aggregator of choice the setup should contain values for [mp_layer, batch_size, neight_dim and hidden_dim] otherwise the process will exit.
            :param features: matrix of feature vectors defining graph verticies
            :param neighbouring: matrix defining graph verticies neighbouring
            :param axis: axis definition if an element-wise aggregator is chosen and for the other matrix operations (default = 1) 
            :param aggregator: aggregator of choice (default = mean)
            :param mp_layer: [max_pool only] => defined max_pooling layer(s) (default = None)
            :param batch_size: [max_pool only] => defined batch size for the pooling layer(s) (default = None)
            :param neight_dim: [max_pool only] => neighbourhood dimension (default = None)
            :param hidden_dim: [max_pool only] => parent hidden layer dimension (default = None)
        """   
        self.initial_features = features
        self.new_features = None
        self.neighbouring = neighbouring
        self.aggregator = aggregator
        self.zero_vec = K.zeros(features[0,:].shape, dtype='float32')
        self.axis = axis
        self.mp_layer = mp_layer
        self.batch_size = batch_size
        self.neight_dim = neight_dim
        self.hidden_dim = hidden_dim

        if (aggregator == 'max_pool'):
            AssertNotNone(self.mp_layer)
            AssertNotNone(self.batch_size)
            AssertNotNone(self.neight_dim)
            AssertNotNone(self.hidden_dim)

    def GetVectorNeighbours(self, features, neighbouring_look_up, index):
        """
        This function collect the neighbourhood vectors for a spezific vertex given by index.
            :param features: list of features defining graph vertices 
            :param neighbouring_look_up: graph neighbourhood look_up
            :param index: desired vertex index
        """   
        AssertNotNegative(index)
        AssertNotNone(features, 'features')
        AssertNotNone(neighbouring_look_up, 'neighbouring look-up')
        neighbouring = neighbouring_look_up[index, :]
        return multiply([features, neighbouring])

    def GetAllVectorsFeatures(self, last_features, in_neighbouring, aggregator='mean'):
        """
        This function collect and aggregates all verticies next hop neighbourhood feature vectors.
            :param last_features: feature list of the previous hop. On first hop it's the initial feature vectors matrix
            :param in_neighbouring: neighbourhood look-up
            :param aggregator: aggregator function definition (default = mean)
        """   
        aggregated_features_vecs = None
        
        for i in range(in_neighbouring.shape[0]):        
            found_neighbour_vectors = self.GetVectorNeighbours(last_features, in_neighbouring, i)
            AssertTensorShapeEqual(last_features, found_neighbour_vectors)
            
            initial_aggregator = Aggregators(found_neighbour_vectors, self.axis, aggregator)
            temp_feats = None

            if (aggregator == 'max_pool'):
                temp_feats = initial_aggregator.PerformAggregator(  mp_layer=self.mp_layer, 
                                                                    batch_size=self.batch_size, 
                                                                    neight_dim=self.neight_dim, 
                                                                    hidden_dim=self.hidden_dim)
            else:
                temp_feats = initial_aggregator.PerformAggregator()

            AssertNotNone(temp_feats, 'temp_features')

            if aggregated_features_vecs is None:
                aggregated_features_vecs = temp_feats
            else: 
                aggregated_features_vecs = K.concatenate([aggregated_features_vecs, temp_feats])
                
        return K.transpose(K.reshape(aggregated_features_vecs, (last_features.shape[0],-1)))

    def Execute(self):
        """
        This function execute the feature aggregation process for the next neighbouring step.
        """   
        AssertIsTensor(self.initial_features)
        AssertIsTensor(self.neighbouring)
        return self.GetAllVectorsFeatures(self.initial_features, 
                                          self.neighbouring,
                                          'mean')
    