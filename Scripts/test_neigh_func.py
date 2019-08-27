import numpy as np
import keras.backend as K
from NetworkHandler.Neighbouring.NeighbourhoodCollector import Neighbourhood as Nhood

class TestNeigh():

    def Execute(self):
        embedding_sz:int = 200
        batch_sz:int = 64
        nodes:int = 35
        aggregator: str ='mean'
        


        feats = K.variable(np.random.rand(batch_sz, nodes, embedding_sz))
        neighs = K.variable(np.random.rand(batch_sz, nodes, nodes))

        print("feats: ", feats.shape)
        print("neighs: ", neighs.shape)

        nhood = Nhood(feats, neighs, aggregator=aggregator, is_2d=False).Execute(batch_sz)
        print("nhood: ", nhood.shape)

if __name__ == "__main__":
    TestNeigh().Execute()