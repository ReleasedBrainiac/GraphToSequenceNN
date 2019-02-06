from NetworkHandler.KerasSupportMethods import KerasEval as KE
from NetworkHandler.KerasSupportMethods import AssertTensorShapeEqual, AssertIsTensor, AssertTensorDotDim, AssertVectorLength, AssertAddTensorToTensor
from keras import backend as K

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

    def AssertNeighbourMatch(self, found_neighbours, neighbouring_look_up):
        check = len(found_neighbours) == KE(K.sum(neighbouring_look_up))
        assert (check), ('Amount of existing neigbours did not match the found neigbours count!')
        return check

    def MaxPoolAggregator(self, vectors, axis=1):
        AssertIsTensor(vectors)
        return K.max(vectors, axis)

    def MeanAggregator(self, vectors, axis=1):
        AssertIsTensor(vectors)
        return K.mean(vectors, axis)

    def PerformAggregator(self, found_vectors, features_shape, axis=1, aggregator='mean'):
        aggregated = None
        
        if (aggregator=='mean'):
            aggregated = self.MeanAggregator(found_vectors, axis)
        else:
            aggregated = self.MaxPoolAggregator(found_vectors, axis)
                
        AssertTensorShapeEqual(aggregated, features_shape)
        return aggregated

    def GetSingleVectorFeatures(self, max_neighbours, vertex_neighbours, last_features):
        found_neigbours = []
        
        if KE(K.sum(vertex_neighbours)) > 0:
            for k in range(max_neighbours):
                    if vertex_neighbours[k].eval() > 0.999999:
                        found_neigbours.append(last_features[k, :])

            self.AssertNeighbourMatch(found_neigbours, vertex_neighbours)
            
        return found_neigbours


    def GetAllVectorsFeatures(self, iterator, last_features, in_neighbouring, axis=1, aggregator='mean'):
        updated_features = []
        for i in range(iterator):
            vertex_neighbours = in_neighbouring[i,:]
            max_neighbours = vertex_neighbours.shape[0]
            found_neigbours = self.GetSingleVectorFeatures(max_neighbours, vertex_neighbours, last_features)
            
            if len(found_neigbours) < 1: found_neigbours.append(self.zero_vec)
            
            #TODO hier muss der max pool oder mean hin da hier dir neig_vecs gefunden und bearbeitet werden
            found_vectors = K.variable(value=found_neigbours, dtype='float32')
            updated_features.append(self.PerformAggregator(found_vectors, last_features[0,:], axis, aggregator))
            
            
        final_features = K.stack(updated_features)
        AssertTensorShapeEqual(last_features, final_features)
        return final_features


    def Execute(self):
        AssertIsTensor(self.initial_features)
        AssertIsTensor(self.neighbouring)
        AssertTensorDotDim(self.initial_features, self.neighbouring)
        self.new_features = self.GetAllVectorsFeatures( self.neighbouring.shape[0],
                                                        self.initial_features, 
                                                        self.neighbouring,
                                                        self.axis,
                                                        self.aggregator)

    def OverwriteNewFeatures(self, new_features):
        assert (new_features != None), ('Given feature vectors are None!')
        AssertIsTensor(new_features)
        self.new_features = new_features
        AssertIsTensor(self.new_features)

    def PerformMaxPoolLayersCall(self, mp_layers, neigh_vecs, batch_size, max_neighbours):
        h_reshaped = K.reshape(neigh_vecs, (batch_size * max_neighbours, self.neigh_input_dim))
        for l in self.mp_layers: h_reshaped = l(h_reshaped)

        neigh_h = K.reshape(h_reshaped, (batch_size, max_neighbours, self.hidden_dim))