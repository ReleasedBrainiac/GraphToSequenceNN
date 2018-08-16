# - *- coding: utf- 8*-

import numpy as np
import tensorflow as tf

def GetMaxColIndexFromTensor(tensor_x):
    count = tensor_x.get_shape()[0]
    return tf.convert_to_tensor(count)

def KerasGetMaxColCountTensor(tensor_x):
    count = tf.keras.backend.shape(tensor_x)[0]
    return tf.keras.backend.variable(value=count, dtype='int32')

def GetMaxRowsIndexFromTensor(tensor_x):
    cols = tensor_x.get_shape()[1]
    return tf.convert_to_tensor(cols)

def GetNoneZerosOfVecTensor(vec_tens):
    return tf.convert_to_tensor(tf.count_nonzero(vec_tens, dtype=tf.float32))   

def GetSummedNeighboursByEdges(neigh_emb, feature_tensor):
    #return tf.matmul(tf.expand_dims(neigh_emb,0), feature_tensor)
    return tf.reduce_sum(tf.multiply(tf.expand_dims(neigh_emb,-1), feature_tensor), axis=0)

# This function return the tensor rank as tensor constant
def GetMaxRankFromTensor(tensor_x):
    return tf.constant(tf.rank(tensor_x).eval())

# This function return the tensors shape as list/narray
def GetShapeListFromTensor(tensor_x):
    return tensor_x.get_shape().as_list()

# This function returns the given tensor and its transposed matrix
# This means we get the next nodes and prev nodes tensor for each node
def GetNextAndPrev(tensor_x):
    next_nodes = tensor_x
    prev_nodes = tf.transpose(tensor_x)
    return tf.convert_to_tensor(next_nodes), tf.convert_to_tensor(prev_nodes)

def GetKerasNAP(tensor_x):
    next_nodes = tf.keras.backend.variable(value=tensor_x, dtype='float32')
    prev_nodes = tf.keras.backend.variable(value=tf.transpose(tensor_x), dtype='float32')
    return next_nodes, prev_nodes

def KerasEval(tensor_x):
    return tf.keras.backend.eval(tensor_x)

def KerasShape(tensor_x):
    return tf.keras.backend.shape(tensor_x)
    
def GetNeighbourEmbeddingWithMean(edge_emb, feature_vec_tensor):
    sum_tensor = GetSummedNeighboursByEdges(edge_emb, feature_vec_tensor)
    divider = GetNoneZerosOfVecTensor(edge_emb)
    return GetMeanOrNull(edge_emb, sum_tensor, divider)

def GetMeanOrNull(edge_emb, sum_tensor, divider):
    return tf.cond(tf.count_nonzero(edge_emb) > 0, 
                   lambda: tf.divide(sum_tensor, divider),
                   lambda: sum_tensor)

def GetDefaultFloat32TensArray():
    return tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)

# This function show the Tensors:
# Data
# Tansor definition
# Shape
# and Range
def ShowSetupInfo(tensor_x):
    tensor_x_range = tf.rank(tensor_x)

    print('#########################')
    print('Tensor Values: \n', tensor_x.eval())
    print('Tensor: ', tensor_x)
    print('Tensor Shape: ', GetShapeListFromTensor(tensor_x))
    print('Tensor Rank: ',  GetMaxRankFromTensor(tensor_x).eval())
    print('Tensor Cols: ',  GetMaxColIndexFromTensor(tensor_x).eval())
    print('Tensor Rows: ',  GetMaxRowsIndexFromTensor(tensor_x).eval())
    print('#########################\n')

    return tensor_x

def GetNextNodeFeatureTensors(edge_emb, node_emb, result_emb, iterator):
    
    #Condition => if current iteration is lower as max iterations
    def cond(edge_emb, node_emb, result_emb, i):
         return tf.less(i, iterator)

    #Loop Body Processing Unit => get all connection dependencies
    def body(edge_emb, node_emb, result_emb, i):
        
        #Get current viewed Node Neighbour embedding mean as result!
        result = GetNeighbourEmbeddingWithMean(edge_emb[i, :], node_emb)
        
        #Store result in Tensorarray result_emb
        result_emb = result_emb.write(i, [result])
        
        #Add +1 to Iterator and return process result!
        return [edge_emb, node_emb, result_emb, tf.add(i, 1)]

    #return the loop setup
    return tf.while_loop(cond, body, [edge_emb, node_emb, result_emb, 0])

def GetTensorResults(edges, nodes, result_tens_arry, iterator):
    res = GetNextNodeFeatureTensors(edges, nodes, result_tens_arry, iterator)
    return res[0], res[1], res[2].concat(), res[3]

def GetFeaturesTensorOnly(edge_and_nodes):
    iterations = KerasGetMaxColCountTensor(edge_and_nodes[0])
    carrier = GetDefaultFloat32TensArray()
    result = GetTensorResults(edge_and_nodes[0], edge_and_nodes[1], carrier, iterations)[2]
    print(result)
    return result 
