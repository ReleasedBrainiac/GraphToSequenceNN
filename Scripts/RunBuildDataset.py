from DatasetHandler.DatasetProvider import DatasetPipeline
from GloVeHandler.GloVeDatasetParser import GloVeEmbedding

#===========================================================#
#                            Test                           #
#===========================================================#
inpath = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
glove_file_path = '../../Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
embedding_out = 100

pipe = DatasetPipeline(in_path=inpath, 
                       output_path_extender='dc.ouput', 
                       max_length=-1, 
                       show_feedback=False, 
                       saving=False, 
                       keep_edges=False
                       )

datapairs = pipe.ProvideData()
node_context_values = []
for datapair in datapairs: node_context_values.append(datapair[1])

embedding = GloVeEmbedding(nodes_context=node_context_values,glove_file_path=glove_file_path, output_dim=embedding_out, use_whole_glove_ww=True).GetGloveEmbeddingLayer()