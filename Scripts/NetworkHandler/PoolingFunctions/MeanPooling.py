import tensorflow as tf

'''
    This class is based on "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks" by Kun Xu et al.  
    The paper can be found at: https://arxiv.org/abs/1804.00823
    The original implementation snipped from the IBM research team can be found at: https://github.com/IBM/Graph2Seq/blob/master/main/pooling.py

    Some smaller changes may depend on the structure of my data or my initial network implementation strategy.
'''

class MeanPooling:

    def mean_pool(self, input_tensor, sequence_length=None):
        """
        Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
        over the last dimension of the input.
        For example, if the input was the output of a LSTM of shape
        (batch_size, sequence length, hidden_dim), this would
        calculate a mean pooling over the last dimension (taking the padding
        into account, if provided) to output a tensor of shape
        (batch_size, hidden_dim).
        Parameters
        ----------
        input_tensor: Tensor
            An input tensor, preferably the output of a tensorflow RNN.
            The mean-pooled representation of this output will be calculated
            over the last dimension.
        sequence_length: Tensor, optional (default=None)
            A tensor of dimension (batch_size, ) indicating the length
            of the sequences before padding was applied.
        Returns
        -------
        mean_pooled_output: Tensor
            A tensor of one less dimension than the input, with the size of the
            last dimension equal to the hidden dimension state size.
        """
        with tf.name_scope("mean_pool"):
            # shape (batch_size, sequence_length)
            input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

            # If sequence_length is None, divide by the sequence length
            # as indicated by the input tensor.
            if sequence_length is None: sequence_length = tf.shape(input_tensor)[-2]

            # Expand sequence length from shape (batch_size,) to
            # (batch_size, 1) for broadcasting to work.
            expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1), "float32") + 1e-08

            # Now, divide by the length of each sequence.
            # shape (batch_size, sequence_length) and return the result.
            return (input_tensor_sum / expanded_sequence_length)