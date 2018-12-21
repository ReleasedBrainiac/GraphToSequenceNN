'''
Resource:
    1. https://github.com/GPflow/GPflow/issues/439 @ javdrher commented on 5 Jul 2017
'''

import tensorflow as tf
import numpy as np
from LambdaNodeEmbedding import KerasEval

def tf_tril_indices(N, k=0):
   M1 = tf.tile(tf.expand_dims(tf.range(N), axis=0), [N,1])
   M2 = tf.tile(tf.expand_dims(tf.range(N), axis=1), [1,N])
   mask = (M1-M2) >= -k
   ix1 = tf.boolean_mask(M2, tf.transpose(mask))
   ix2 = tf.boolean_mask(M1, tf.transpose(mask))
   return ix1, ix2

with tf.Session() as sess:
    X = tf.ones([4, 4])
    
    # Standard
    ix1, ix2 = tf_tril_indices(tf.shape(X)[0])
    sess.run(tf.global_variables_initializer())
    
    print ('Start_1:\n', sess.run([ix1, ix2]))
    print(tf.keras.backend.eval(X))
    print(np.tril_indices(4) [0])
    
    # With diagonal offset
    ix1, ix2 = tf_tril_indices(tf.shape(X)[0], 1)
    sess.run(tf.global_variables_initializer())
    print ('Start_2:\n', sess.run([ix1, ix2]))
    print( np.tril_indices(4, 1) )