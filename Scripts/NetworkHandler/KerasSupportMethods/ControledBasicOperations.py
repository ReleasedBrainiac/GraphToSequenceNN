from keras import backend as K
from NetworkHandler.KerasSupportMethods import AssertIsTensor, AssertTensorDotDim, AssertAddTensorToTensor

'''
    This class is based on https://keras.io/backend/#backend-functions.
    This implementation is refined for Keras usage.
'''

class ControledTensorOperations():
    """
    This class allow to perform various basic Keras operations in addition with some control statements!
    """

    def ControledConcatenation(initial_tensor, concat_tensor, axis=1):
        """
        This function perform a concatenation of 2 tensors.
            :param initial_tensor: initial tensor
            :param concat_tensor: tensor to be concatenated to the initial tensor
            :param axis: concatenation axis (default = 1)
        """
        AssertIsTensor(initial_tensor)
        AssertIsTensor(concat_tensor)
        return K.concatenate([initial_tensor, concat_tensor], axis)

    def ControledWeightDotProduct(matrix_left, matrix_top):
        """
        This function perform the algebraic tensor matrix product between matrix_left and matrix_top.
            :param matrix_left: the first tensor matrix
            :param matrix_top: the second tensor matrix
        """
        AssertIsTensor(matrix_left)
        AssertIsTensor(matrix_top)
        AssertTensorDotDim(matrix_left, matrix_top)
        return K.dot(matrix_left, matrix_top)

    def ControledBiased(tensor, bias):
        """
        This function add a bias to a given tensor matrix.
            :param tensor: the tensor matrix
            :param bias: the desired bias
        """   
        AssertIsTensor(tensor)
        AssertIsTensor(bias)
        AssertAddTensorToTensor(tensor, bias) 
        return tensor + bias