import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dense

class BahdanauAttention(keras.Model):
    """
    This class implements the BahdanauAttention mechanism.

    !!!! ATTENTION !!!! 
    The Code is directly used from the tesnroflow tutorials -> https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention 
    The only addition are the try/catch blocks.
    """
    def __init__(self, units):
        try:
            super(BahdanauAttention, self).__init__()
            self.W1 = Dense(units)
            self.W2 = Dense(units)
            self.V = Dense(1)
        except Exception as ex:
            template = "An exception of type {0} occurred in [BahdanauAttention.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def __call__(self, sample_outs, hidden_state):
        """
        The call method of the (keras) layer.
            :param sample_outs:
            :param hidden_state:  
        """
        try:
            # we are doing this to perform addition to calculate the score
            hidden_with_time_axis = K.expand_dims(hidden_state, axis=1)

            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(K.tanh(self.W1(sample_outs) + self.W2(hidden_with_time_axis)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = K.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * sample_outs
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights
        except Exception as ex:
            template = "An exception of type {0} occurred in [BahdanauAttention.call]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)