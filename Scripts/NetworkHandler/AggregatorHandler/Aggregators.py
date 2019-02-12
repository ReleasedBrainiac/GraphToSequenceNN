from keras import backend as K
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertIsTensor, AssertNotNone

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    The original implementation snippeds from the IBM research team implemented in tensorflow can be found at:
        1. https://github.com/IBM/Graph2Seq/blob/master/main/aggregators.py
        2. https://github.com/IBM/Graph2Seq/blob/master/main/pooling.py

    Attention: 
        1. The mean calc is implemented like the Keras GlobalAveragePooling1D (without masked sum masking)!
        2. The max pooling is implemented like the Keras GlobalMaxPooling1D 
        => https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557
        => https://keras.io/layers/pooling/
'''

class Aggregators():
    """
    This class provide the mean and max_pool aggregation functions.
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
        This function return  the element-wise max of the given vectors depending on the selected axis (=> step_axis).
        """   
        return K.max(self.vectors, self.axis)

    def MeanAggregator(self):
        """
        This function return  the element-wise mean of the given vectors depending on the selected axis (=> step_axis).
        """
        return K.mean(self.vectors, self.axis)

    def PerformAggregator(self):
        """
        This funtion perform the selected aggregation.
        """   
        if (self.aggregator=='max_pool'):
            return self.MaxAggregator()
        else:
            return self.MeanAggregator()