from NetworkHandler.KerasSupportMethods.SupportMethods import AssertTensorShapeEqual, AssertIsTensor, AssertNotNone, AssertNotNegative, IsKerasTensor
from NetworkHandler.AggregatorHandler.Aggregators import Aggregators
from keras import backend as K
from keras.layers import multiply

class Neighbourhood():
    """
    This class calculates the neighborhood feature aggregation.

    This class is inspired by "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    The Keras resource can found here:
        1. https://keras.io/layers/merge/#multiply_1
        2. https://keras.io/backend/#backend
    """

    def __init__(self, features, neighbouring, axis:int=1, aggregator:str ='mean'):
        """
        This constructor stores all necessary variables for the setup.
            :param features: matrix of feature vectors defining graph verticies
            :param neighbouring: matrix defining graph verticies neighbouring
            :param axis:int: axis definition if an element-wise aggregator is chosen and for the other matrix operations (default = 1) 
            :param aggregator:str: aggregator of choice (default = mean)
        """   
        self.features = features
        self.neighbouring = neighbouring
        self.aggregator = aggregator
        self.axis = axis

    def GetVectorNeighbours(self, index:int):
        """
        This function collects the neighbourhood vectors for a spezific vertex given by index.
            :param index:int: desired vertex index
        """
        try:
            AssertNotNegative(index)
            AssertNotNone(self.features, 'features')
            AssertNotNone(self.neighbouring, 'neighbouring look-up')
            neighbouring = self.neighbouring[index, :]
            neighbouring = K.reshape(neighbouring, (neighbouring.shape[0],-1))
            return multiply([self.features, neighbouring])
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.GetVectorNeighbours]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def GetAllVectorsFeatures(self):
        """
        This function collects and aggregates all verticies next hop neighbourhood feature vectors.
        """
        try:
            agg_f_vecs = None
            vecs = self.neighbouring.shape[1]

            for i in range(vecs):        
                found_neighbour_vectors = self.GetVectorNeighbours(i)       
                aggregator_result = Aggregators(found_neighbour_vectors, self.axis, self.aggregator).Execute()
                AssertNotNone(aggregator_result, 'aggregator_result')
                agg_f_vecs = aggregator_result if (agg_f_vecs is None) else K.concatenate([agg_f_vecs, aggregator_result])

            transpose = K.transpose(K.reshape(agg_f_vecs, (vecs,-1)))
            return K.concatenate([self.features,transpose])
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.GetAllVectorsFeatures]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def Execute(self):
        """
        This function executes the feature aggregation process for the next neighbouring step.
        """   
        try:
            AssertIsTensor(self.features)
            AssertIsTensor(self.neighbouring)
            return self.GetAllVectorsFeatures()
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 
