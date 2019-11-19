import numpy as np
import keras.backend as K
from keras.layers import Input
import time
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood
from NetworkHandler.Builder import ModelBuilder

class TestNeigh():

    def Execute(self):
        t0 = time.clock()
        embedding_sz:int = 200
        batch_sz:int = 16
        nodes:int = 35
        aggregator: str ='mean'
        


        feats = K.variable(np.random.rand(batch_sz, nodes, embedding_sz))
        neighs = K.variable(np.random.rand(batch_sz, nodes, nodes))

        print("feats: ", feats.shape)
        print("neighs: ", neighs.shape)

        nhood = Nhood(feats, neighs, aggregator=aggregator, is_2d=False).Execute(batch_sz)
        print("nhood: ", nhood.shape)
        t1 = time.clock()

        print("Process Time: ", t1-t0)

    def TestBahdanau(self):

        dummy_hidden = K.zeros((64, 1024))
        dummy_outs = K.zeros((64, 16, 1024))

        print ('Encoder output shape: (batch size, sequence length, units) {}'.format(dummy_outs.shape))
        print ('Encoder hidden state shape: (batch size, units) {}'.format(dummy_hidden.shape))


        mb = ModelBuilder.ModelBuilder( input_enc_dim = 1, 
                                        edge_dim = 1, 
                                        input_dec_dim = 1, 
                                        batch_size = 64)

        attention_result, attention_weights = mb.BuildBahdanauAttentionPipe(10, dummy_outs, dummy_hidden)

        print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
        print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

if __name__ == "__main__":
    TestNeigh().TestBahdanau()