from keras import backend as K
from NetworkHandler.KerasSupportMethods.SupportMethods import AssertIsTensor, AssertTensorDotDim, AssertNotNegative, AssertAddTensorToTensor, AssertNotNone

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

    def ControlledMatrixExtension(tensor, vector, times):
        """
        This function extends a tensor with tensor along axis 1.
            :param tensor: extendable tensor
            :param vector: a vector matching tensor on shape[0]
            :param times: number of times the zeros should be concatenated to the tensor
        """   
        AssertNotNone(tensor, 'Extendable tensor')
        AssertNotNone(vector, 'Concatenating vector')
        AssertNotNegative(times)

        if times != 0:
            vector = K.reshape(vector, (vector.shape[0],1))
            assert (tensor.shape[0] == vector.shape[0]), ('Vector dim',vector.shape[0],'mismatch tensor dim',tensor.shape[0],'on concatenation axis')
        
            for i in range(times):
                tensor = ControledTensorOperations.ControledConcatenation(tensor, vector)
        
            AssertNotNone(tensor, 'Extension result')
        return tensor

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

    def ControledShapeDifference(kernel_tensor, tensor, index_kernel=0, index_other=1):
        """
        This function allow to calculate the difference between 2 tensors desired shape values.
            :param kernel_tensor: first tensor for shape diff calculation
            :param tensor: second tensor for shape diff calculation
            :param index_kernel: index for kernel tensor shape value selection (Default = 0)
            :param index_other: index for second tensor shape value selection (Default = 1)
        """   
        AssertIsTensor(kernel_tensor)
        AssertIsTensor(tensor)
        assert (kernel_tensor.shape[index_kernel] >= tensor.shape[index_other]), ('Kernel',index_kernel,'-dimension is lower then concatenation',index_other,'-dimension!')
        return kernel_tensor.shape[index_kernel] - tensor.shape[index_other]