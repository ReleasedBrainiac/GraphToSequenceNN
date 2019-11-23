import numpy as np
import keras.backend as K
from keras.layers import Lambda, Embedding, LSTM, concatenate,GRU
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

        # From = > https://www.tensorflow.org/tutorials/text/nmt_with_attention

        vocab_size = 15000
        embedding_dim = 400
        words = 64
        batch_size = 32
        dummy_hidden = K.zeros((32, embedding_dim))
        dummy_outs = K.zeros((32, 16, embedding_dim))
        embeddings = K.zeros((32, words))

        print ('Encoder output shape: (batch size, sequence length, units) {}'.format(dummy_outs.shape))
        print ('Encoder hidden state shape: (batch size, units) {}'.format(dummy_hidden.shape))


        mb = ModelBuilder.ModelBuilder( input_enc_dim = 1, 
                                        edge_dim = 1, 
                                        input_dec_dim = 1, 
                                        batch_size = batch_size)

        context_vector, attention_weights = mb.BuildBahdanauAttentionPipe(embedding_dim, dummy_outs, dummy_hidden)
        print("Attention result shape 1: (batch size, units) {}".format(context_vector.shape))
        print("Attention weights shape 1: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

        #context_vector = Lambda(lambda q: K.expand_dims(q, axis=1), name="attention_reshape")(context_vector)
        #print("Attention result shape 2: (batch size, units) {}".format(context_vector.shape))

        embedding = Embedding(vocab_size, embedding_dim)(embeddings)
        print("Embedding result shape 1: (batch size, units) {}".format(embedding.shape))
        print("Embedding act units: ", context_vector.shape[-1].value)

        #embedding = LSTM(context_vector.shape[-1].value, batch_size=batch_size, return_sequences=True, name="attention_activate")(embedding)
        print("Embedding result shape 2: (batch size, units) {}".format(embedding.shape))

        context_vector = Lambda(lambda q: K.expand_dims(q, axis=1), name="attention_reshape")(context_vector)
        print("Attention result shape 2: (batch size, units) {}".format(context_vector.shape))

        stated_att_encoder = concatenate([context_vector,embedding], name="att_emb_concatenation", axis=-1)
        print("Stated Encoder result shape: (batch size, units) {}".format(stated_att_encoder.shape))

if __name__ == "__main__":
    TestNeigh().TestBahdanau()