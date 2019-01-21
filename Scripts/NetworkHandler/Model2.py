'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    A implementation example of this paper from the IBM research team can be found at: https://github.com/IBM/Graph2Seq 

    This model implementation is my own interpretation of the provided structure in tht paper.
'''

from keras import backend as K
from keras import initializers, activations, regularizers
from keras.layers import Layer, Dense, Activation, Input
from keras.models import Model

class CustomGraphToSequenceModel(object):

    SUB_MODEL_TYPES = ['forward', 'backward']

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 name='kernel_weights',
                 weight_init='glorot_uniform',
                 dropout=0.,
                 activation=activations.relu, 
                 placeholders=None, 
                 bias=True, 
                 bias_init='zeros',
                 featureless=False,
                 sparse_inputs=False, 
                 **kwargs):

        #self.forward_sub_model = self.SingleForwardLayerPairs()
        #self.backward_sub_model = self.SingleBackwardLayerPairs()

        self.forward_node_tensor = backward_input_tensor = Input(   shape=(784,),
                                                                    batch_shape=(10,784),
                                                                    name='backward_tensor',
                                                                    dtype='float32',
                                                                    sparse=True)

        self.backward_node_tensor = backward_input_tensor = Input(   shape=(784,),
                                                                     batch_shape=(10,784),
                                                                     name='backward_tensor',
                                                                     dtype='float32',
                                                                     sparse=True)

    



    def BuildHopDefinedSubModel(self, hop_size=1, sub_model_type='forward'):
        if sub_model_type == 
            for hop in range(hop_size):

    	
    def SingleForwardLayerPairs(self):
        pass

    def SingleBackwardLayerPairs(self):
        pass



    def Aggregator(parameter_list):
        pass

    def funcname(parameter_list):
        pass
