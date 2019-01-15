import tensorflow as tf
from inits import zeros
import configure as conf

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/layers.py
    
    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
        Implementation inspired by keras (http://keras.io).
        # Properties
            name: String, defines the variable scope of the layer.
            logging: Boolean, switches Tensorflow histogram logging on/off
        # Methods
            _call(inputs): Defines computation graph of layer
                (i.e. takes input, returns output)
            __call__(inputs): Wrapper for _call()
        """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs