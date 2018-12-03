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

embedding = GloVeEmbedding(nodes_context=node_context_values, vocab_size=400000 ,glove_file_path=glove_file_path, output_dim=embedding_out, use_whole_glove_ww=True).GetGloveEmbeddingLayer()

#//TODO 1. Erst natural Encoding für den tokenizer erstellen!

#//TODO 2. Dann Embeddinglayer bereistellen und natural Encoding für Graph übergeben.um Vector encoding zu erhalten

#//TODO 3. Dann Vector encoding dem NodeEmbeddingLAyer übergeben

#//TODO 4. Dann Node embeddings dem Graph embedding layer übergeben.

#//TODO 5. Graph embedding dem Lstm übergeben

#//TODO 6. Autoencoder structure mittels decoder lstm vervollständigen

#//TODO 7. Überachtes training durchführen

#//TODO 8. Prediction case bauen.