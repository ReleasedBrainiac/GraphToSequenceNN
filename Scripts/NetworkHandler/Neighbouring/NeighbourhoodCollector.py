from NetworkHandler.KerasSupportMethods import AssertTensorShapeEqual, AssertIsTensor, AssertNotNone, AssertNotNegative
from keras import backend as K
from keras.layers import multiply


#TODO missing docu
#TODO missing reference to resource

class NeighbourAggregator():

    def __init__(self, features, neighbouring, axis=1, aggregator='mean',**kwargs):
        self.initial_features = features
        self.new_features = None
        self.neighbouring = neighbouring
        self.aggregator = aggregator
        self.zero_vec = K.zeros(features[0,:].shape, dtype='float32')
        self.axis = axis

    def GetVectorNeighbours(self, features, neighbouring_look_up, index):
        AssertNotNegative(index)
        AssertNotNone(features, 'features')
        AssertNotNone(neighbouring_look_up, 'neighbouring look-up')
        neighbouring = neighbouring_look_up[index, :]
        return multiply([features, neighbouring])

    def GetAllVectorsFeatures(self, last_features, in_neighbouring, axis=0, aggregator='mean'):
        aggregated_features_vecs = None
        
        for i in range(in_neighbouring.shape[0]):        
            found_neighbour_vectors = self.GetVectorNeighbours(last_features, in_neighbouring, i)
            AssertTensorShapeEqual(last_features, found_neighbour_vectors)
            
            if aggregated_features_vecs is None:
                aggregated_features_vecs = PerformAggregator(found_neighbour_vectors, 1, aggregator)
            else: 
                next_result = PerformAggregator(found_neighbour_vectors, 1, aggregator)
                aggregated_features_vecs = K.concatenate([aggregated_features_vecs, next_result])
            
        return K.transpose(K.reshape(aggregated_features_vecs, (last_features.shape[0],-1)))

    def Execute(self, initial_features, in_neighbouring):
        AssertIsTensor(self.initial_features)
        AssertIsTensor(self.neighbouring)
        aggregated_feature_vecs = self.GetAllVectorsFeatures(self.initial_features, 
                                                             self.neighbouring,
                                                             0,
                                                             'mean')
        return aggregated_feature_vecs

    def OverwriteNewFeatures(self, new_features):
        assert (new_features != None), ('Given feature vectors are None!')
        AssertIsTensor(new_features)
        self.new_features = new_features
        AssertIsTensor(self.new_features)

    def PerformMaxPoolLayersCall(self, mp_layers, neigh_vecs, batch_size, max_neighbours):
        h_reshaped = K.reshape(neigh_vecs, (batch_size * max_neighbours, self.neigh_input_dim))
        for l in self.mp_layers: h_reshaped = l(h_reshaped)

        neigh_h = K.reshape(h_reshaped, (batch_size, max_neighbours, self.hidden_dim))