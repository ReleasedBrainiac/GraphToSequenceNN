from keras import backend as K
from NetworkHandler.KerasSupportMethods import AssertIsTensor, AssertNotNone

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    The original implementation snippeds from the IBM research team implemented in tensorflow can be found at:
        1. https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py
        2. https://github.com/IBM/Graph2Seq/blob/master/main/pooling.py
'''

#TODO missing docu

class Aggregators():
    """
    This class provide all aggregation functions 
    """
    
    def __init__(self, vectors, axis=0, aggregator='mean'):
        """
        This constructor collect the general setup for the aggregators.
        Spezific values will be collected at the PerformAggregator function.
        Possible aggregators:
             =>[mean, max, max_pool]
            :param vectors: tensor of found neighbourhood vectors
            :param axis=0: defined axis for element-wise mean or max
            :param aggregator: desired neighbourhood aggregator (default = 'mean')
        """ 
        AssertIsTensor(vectors)
        self.vectors = vectors
        self.axis = axis
        self.aggregator = aggregator

    def MaxAggregator(self):
        """
        This function return  the element-wise max of the given vectors depending on the selected axis.
        """   
        return K.max(self.vectors, self.axis)

    def MaxPoolAggregator(self, mp_layers, batch_size, neight_dim, hidden_dim):
        """
        This function calculates the max pooling of the given vectors by using pooling layers.
        This function is completely collected from the referenced resources!
        If this won't work issue at the referenced resource.
            :param mp_layers: the prepared pooling layers
            :param batch_size: batch size
            :param neight_dim: dimension of neighbourhood vectors
            :param hidden_dim: dimension of the hidden layer
        """   
        max_neighbours = self.vectors.shape[0]
        h_reshaped = K.reshape(self.vectors, (batch_size * max_neighbours, neight_dim))

        for l in self.mp_layers: 
            h_reshaped = l(h_reshaped)

        return K.reshape(h_reshaped, (batch_size, max_neighbours, hidden_dim))

    def MeanAggregator(self):
        """
        This function return  the element-wise mean of the given vectors depending on the selected axis.
        """
        return K.mean(self.vectors, self.axis)

    def PerformAggregator(self, mp_layer=None, batch_size=None, neight_dim=None, hidden_dim=None):
        """
        This funtion perform the selected aggregation.
        """   
        if (self.aggregator=='max'):
            return self.MaxAggregator()
        elif (self.aggregator=='max_pool'):
            AssertNotNone(mp_layer)
            AssertNotNone(batch_size)
            AssertNotNone(neight_dim)
            AssertNotNone(hidden_dim)
            return self.MaxPoolAggregator(mp_layer, batch_size, neight_dim, hidden_dim)
        else:
            return self.MeanAggregator()