from keras import backend as K
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertIsTensor

class Aggregators:
    """
    This class provide the mean and max aggregation functions.

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    Resources
        => https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557
        => https://keras.io/layers/pooling/
    """
    
    def __init__(self, vectors, axis:int =0, aggregator:str ='mean'):
        """
        This constructor collect the general setup for the aggregators.
        Spezific values will be collected at the PerformAggregator function.
        Possible aggregators:
             =>[mean, max]
            :param vectors: tensor of found neighbourhood vectors
            :param axis:int =0: defined axis for element-wise mean or max
            :param aggregator:str: desired neighbourhood aggregator (default = 'mean')
        """ 
        AssertIsTensor(vectors)
        self.vectors = vectors
        self.axis = axis
        self.aggregator = aggregator

    def MaxAggregator(self):
        """
        This function return  the element-wise max of the given vectors depending on the selected axis (=> step_axis).
        """
        try:
            return K.max(self.vectors, self.axis)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Aggregators.MaxAggregator]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def MeanAggregator(self):
        """
        This function return  the element-wise mean of the given vectors depending on the selected axis (=> step_axis).
        """
        try:
            return K.mean(self.vectors, self.axis)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Aggregators.MeanAggregator]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def Execute(self):
        """
        This funtion perform the selected aggregation.
        """   
        try:
            if (self.aggregator=='max'):
                return self.MaxAggregator()
            else:
                return self.MeanAggregator()
        except Exception as ex:
            template = "An exception of type {0} occurred in [Aggregators.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 