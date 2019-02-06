from keras import backend as K

#TODO missing docu
#TODO missing reference to resource

class Aggregators():

    def __init__(self, vectors, axis=0, aggregator='mean'):
        self.vectors = vectors
        self.axis = axis
        self.aggregator = aggregator

    def MaxAggregator(self):
        return K.max(self.vectors, self.axis)

    def MaxPoolAggregator(self):
        return K.max(self.vectors, self.axis)

    def MeanAggregator(self):
        return K.mean(self.vectors, self.axis)

    def PerformAggregator(self):
        aggregated = None
        
        if (self.aggregator=='mean'):
            aggregated = self.MeanAggregator()
        elif (self.aggregator=='max_pool'):
            aggregated = self.MeanAggregator()
        else:
            aggregated = self.MaxAggregator()
                
        return aggregated