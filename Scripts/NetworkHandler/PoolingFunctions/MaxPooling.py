import tensorflow as tf

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/pooling.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

class MaxPooling:

    def handle_pad_max_pooling(self, input_tensor, last_dim):
        """
        docstring here
            :param self: 
            :param input_tensor: 
            :param last_dim: 
        """   
        input_tensor = tf.reshape(input_tensor, [-1, last_dim])
        bs = tf.shape(input_tensor)[0]
        tt = tf.fill(tf.stack([bs, last_dim]), -1e9)
        cond = tf.not_equal(input_tensor, 0.0)
        return tf.where(cond, input_tensor, tt)

    def max_pool(self, input_tensor, last_dim, sequence_length=None):
        """
        Given an input tensor, do max pooling over the last dimension of the input
        :param input_tensor:
        :param sequence_length:
        :return:
        """
        with tf.name_scope("max_pool"):
            #shape [batch_size, sequence_length]
            mid_dim = tf.shape(input_tensor)[1]
            input_tensor = self.handle_pad_max_pooling(input_tensor, last_dim)
            input_tensor = tf.reshape(input_tensor, [-1, mid_dim, last_dim])
            input_tensor_max = tf.reduce_max(input_tensor, axis=-2)
            return input_tensor_max