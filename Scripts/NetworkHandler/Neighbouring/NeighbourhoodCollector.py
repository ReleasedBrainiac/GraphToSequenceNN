from NetworkHandler.KerasSupportMethods.SupportMethods import AssertTensorShapeEqual, AssertIsTensor, AssertNotNone, AssertNotNegative, IsKerasTensor
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
    def __init__(self, features, neighbouring, axis=1, aggregator='mean'):
        """
        This constructor collect all necessary variables for the setup.
            :param features: matrix of feature vectors defining graph verticies
            :param neighbouring: matrix defining graph verticies neighbouring
            :param axis: axis definition if an element-wise aggregator is chosen and for the other matrix operations (default = 1) 
            :param aggregator: aggregator of choice (default = mean)
        """   
        self.features = features
        self.neighbouring = neighbouring
        self.aggregator = aggregator
        self.axis = axis

    def GetVectorNeighbours(self, index):
        """
        This function collect the neighbourhood vectors for a spezific vertex given by index.
            :param index: desired vertex index
        """   
        AssertNotNegative(index)
        AssertNotNone(self.features, 'features')
        AssertNotNone(self.neighbouring, 'neighbouring look-up')
        neighbouring = self.neighbouring[index, :]
        neighbouring = K.reshape(neighbouring, (neighbouring.shape[0],-1))
        return multiply([self.features, neighbouring])

    def GetAllVectorsFeatures(self):
        """
        This function collect and aggregates all verticies next hop neighbourhood feature vectors.
        """   
        aggregated_features_vecs = None
        vecs = self.neighbouring.shape[1]

        for i in range(vecs):        
            found_neighbour_vectors = self.GetVectorNeighbours(i)       
            aggregator_result = Aggregators(found_neighbour_vectors, self.axis, self.aggregator).PerformAggregator()
            AssertNotNone(aggregator_result, 'aggregator_result')

            if aggregated_features_vecs is None:
                aggregated_features_vecs = aggregator_result
            else: 
                aggregated_features_vecs = K.concatenate([aggregated_features_vecs, aggregator_result])
                
        transpose = K.transpose(K.reshape(aggregated_features_vecs, (vecs,-1)))
        return K.concatenate([self.features,transpose])

    def Execute(self):
        """
        This function execute the feature aggregation process for the next neighbouring step.
        """   
        AssertIsTensor(self.features)
        AssertIsTensor(self.neighbouring)
        return self.GetAllVectorsFeatures()
