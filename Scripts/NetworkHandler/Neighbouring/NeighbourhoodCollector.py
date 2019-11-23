from NetworkHandler.KerasSupportMethods.SupportMethods import AssertIsTensor, AssertNotNone, AssertNotNegative
from DatasetHandler.ContentSupport import isNotNone, isNone
from NetworkHandler.AggregatorHandler.Aggregators import Aggregators
from keras import backend as K
from keras.layers import multiply

class Neighbourhood:
    """
    This class calculates the neighborhood feature aggregation.

    This class is inspired by "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823

    This implementation is refined for Keras usage.
    All changes depend on the structure of my input data, the API of Keras or my initial network implementation strategy.

    The Keras resource can found here:
        1. https://keras.io/layers/merge/#multiply_1
        2. https://keras.io/backend/#backend

    Some problems i also faced!
        1. https://github.com/tensorflow/tensorflow/issues/31991
    """

    def __init__(self, features, neighbouring, batch_sz:int = 32, axis:int=1, aggregator:str ='mean', is_2d:bool =True):
        """
        This constructor stores all necessary variables for the setup.
            :param features: matrix of feature vectors defining graph verticies
            :param neighbouring: matrix defining graph verticies neighbouring
            :param batch_sz:int: desired batch size
            :param axis:int: axis definition if an element-wise aggregator is chosen and for the other matrix operations (default = 1) 
            :param aggregator:str: aggregator of choice (default = mean)
            :param is_2d:bool: relevant if the function is used on single samples. allows the inserted 2D return a 2D result otherwise its 3D
        """
        self.features = features
        self.neighbouring = neighbouring
        self.aggregator = aggregator
        self.axis = axis
        self.is_2d = is_2d
        self.batch_sz:int = batch_sz

        #self.batch_sz:int = K.int_shape(self.features)[0] if isNotNone(self.features.shape[0].value) else batch_sz

    def GetVectorNeighbours(self, features, neighbouring, index:int):
        """
        This function collects the neighbourhood vectors for a spezific vertex given by index.
            :param features: previous features
            :param neighbouring: neighbourhood look up
            :param index:int: viewed feature index
        """
        try:
            AssertNotNegative(index)
            AssertNotNone(features, 'features')
            AssertNotNone(neighbouring, 'neighbouring look-up')

            neighbouring = neighbouring[index, :]
            neighbouring = K.reshape(neighbouring, (neighbouring.shape[0],-1))
            return multiply([features, neighbouring])
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.GetVectorNeighbours]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetSamplesAggregatedFeatures(self, features, neighbourhood, features_size:int):
        """
        This function collects and aggregates all given features next hop neighbourhood feature vectors for 1 sample.
            :param features: samples previous or initial features
            :param neighbourhood: samples graph neighbourhood look up
            :param features_size:int: amount of features in the sample
        """   
        try:
            agg_f_vecs = None
            for i in range(features_size):        
                found_neighbour_vectors = self.GetVectorNeighbours(features, neighbourhood, i)
                aggregator_result = Aggregators(found_neighbour_vectors, self.axis, self.aggregator).Execute()
                AssertNotNone(aggregator_result, 'aggregator_result')

                if (agg_f_vecs is None):
                    agg_f_vecs = [aggregator_result] 
                else: 
                    agg_f_vecs.append(aggregator_result)
                
            agg_f_vecs = agg_f_vecs[0] if (features_size < 2) else K.concatenate(agg_f_vecs)

            return K.transpose(K.reshape(agg_f_vecs, (features_size,-1)))
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.GetSamplesAggregatedFeatures]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def GetAllSamplesAggregatedFeatures(self):
        """
        This function collects and aggregates all given features next hop neighbourhood feature vectors for all samples.
        Attention!
            => since the neighbourhood tensors have to be for each sample an (MxM) matrix i collect the dim from last shape value!
        """   
        try:
            samples_results = []
            dims = len(self.neighbouring.shape)
            vecs = self.neighbouring.shape[dims-1]

            #batch_sz:int = K.int_shape(self.features)[0] if isNotNone(self.features.shape[0].value) else batch_sz

            for sample in range(self.features.shape[0].value):
                
                sample_concatenate = None
                sample_features = self.features[sample,:,:] if (dims > 2) else self.features
                sample_neigbourhood = self.neighbouring[sample,:,:] if (dims > 2) else self.neighbouring

                agg_f_vecs = self.GetSamplesAggregatedFeatures(features=sample_features, neighbourhood=sample_neigbourhood, features_size=vecs)
                sample_concatenate = K.concatenate([sample_features,agg_f_vecs])

                AssertNotNone(sample_concatenate, 'sample_concatenate')
                samples_results.append(sample_concatenate)

            assert samples_results, 'No results calculated for feature and neighbourhood samples!'
            return samples_results
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.GetAllSamplesAggregatedFeatures]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def InputStrategySelection(self):
        """
        This function processes the neighbourhood collection and selects a dimension conversion strategy for the desired returning dimension (2D or 3D)
        """
        try:
            samples_results = self.GetAllSamplesAggregatedFeatures()

            if self.is_2d:
                return samples_results[0]
            else:
                if(self.features.shape[0].value == 1):
                    return K.expand_dims(samples_results[0], axis=0)
                else:  
                    return K.stack(samples_results)
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.InputStrategySelection]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def Execute(self):
        """
        This function executes the feature aggregation process for the next neighbouring step.
        """   
        try:
            AssertIsTensor(self.features)
            AssertIsTensor(self.neighbouring)

            if isNotNone(self.features.shape[0].value):
                print("Normal Shape: ", self.features.shape)
                return self.InputStrategySelection()
            else:
                placeholder = K.zeros(shape=(self.batch_sz, self.neighbouring.shape[1], (self.features.shape[-1] + self.neighbouring.shape[-1])))
                print("PH Shape: ", placeholder.shape)
                return placeholder
        except Exception as ex:
            template = "An exception of type {0} occurred in [NeighbourhoodCollector.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 
