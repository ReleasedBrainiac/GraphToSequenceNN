# - *- coding: utf- 8*-
from LambdaNodeEmbedding import GetKerasNAP, GetKerasInputLayerPairs, GetKerasInputLayerPairs, KerasEval
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda

#These are the edges between the nodes.
edgelist = np.array([
                    [0,1,1,1],
                    [0,0,0,1],
                    [0,1,0,0],
                    [0,0,0,0]
                    ], dtype='float32')

#These are the node features examples
features_init = np.array([
                         [0., 0., 1., 1., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 1., 1., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 1., 0.]
                         ], dtype='float32')

result_next_feat = np.array([
                            [0., 0.6666667, 0., 0., 0.33333334, 0.33333334, 0.33333334, 0.33333334, 0.],
                            [0., 1., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 1., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.]
                            ], dtype='float32')

edgelist_ten = K.variable(value=edgelist, dtype='float32')
fv_init_ten = K.variable(value=features_init, dtype='float32')
result_feat_ten = K.variable(value=result_next_feat, dtype='float32')
next_t, prev_t = GetKerasNAP(edgelist_ten)

test_in = [next_t, fv_init_ten]

print("Test1:", KerasEval(test_in[0]), '\n\n')
print("Test2:", KerasEval(test_in[1]))


###########
#  Build  #
###########

ins = GetKerasInputLayerPairs((None,),"edges",(9,),"features")
x = GetMyKerasLambda(ins,(None,9),"MyLambda")
x = Dense(9, activation="softmax", name="LastDenseLayer")(x)
model = Model(inputs=ins, outputs=x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

###########
#  Run 1  #
###########

model.fit(test_in, 
          [test_in], 
          steps_per_epoch=1, 
          validation_steps=1, 
          epochs=1, 
          verbose=0, 
          shuffle=False)

###########
#  Run 2  #
###########


test_in = [next_t, fv_init_ten]
t1 = [edgelist_ten, fv_init_ten]
t2 = [edgelist, features_init]

model.fit([edgelist_ten, fv_init_ten], 
          t2, 
          steps_per_epoch=1, 
          validation_steps=1, 
          epochs=1, 
          verbose=0, 
          shuffle=False)
