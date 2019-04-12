import numpy as np
from numpy import asarray, zeros
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer


class GloVeEmbedding:
    """
    This class provides the GloVeEmbedding Layer and the conversion of words into vectors of numbers.

    This part of my work is inspired by the code of:
        1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
        2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
        3. https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings 
        4. https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html 

    The GloVe dataset was provided by:
        https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
    """

    GLOVE_DTYPE = 'float32'
    GLOVE_ENC = 'utf8'
    MAX_NUM_WORDS = -1
    NUMBER_WORDS = -1

    def __init__(self, 
                 tokenizer:Tokenizer, 
                 max_cardinality:int =-1, 
                 vocab_size:int =20000, 
                 max_sequence_length:int =1000, 
                 glove_file_path:str = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt', 
                 output_dim:int =100,
                 batch_size:int = 1,
                 show_feedback:bool =False):
        """
        This constructor stores all given parameters. 
        Further it loads the word embedding indices from the datasets node dictionairies.
        The output_dim values should be adapted from the correpsonding GloVe file definiton [100d, 200d or 300d].
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param tokenizer:Tokenizer: tokenizer from GloVe dataset preprocessing.
            :param max_cardinality:int: the max graph node cardinality in the dataset
            :param vocab_size:int: maximum number of words to keep, based on word frequency
            :param max_sequence_length:int: max length length over all sequences (padding)
            :param glove_file_path:str: path of the desired GloVe word vector file
            :param output_dim:int: the general vector size for each word embedding
            :param batch_size:int: embedding batch size
            :param show_feedback:bool: switch allows to show process response on console or not
        """   
        try:
            print('~~~~~~~~ Init Embedding GloVe ~~~~~~~~~') 
            self.GLOVE_DIR = glove_file_path
            print('GloVe file:\t\t=> ', self.GLOVE_DIR)

            self.EMBEDDING_DIM = output_dim if (output_dim > 0) else 100
            print('Output dimension:\t=> ', self.EMBEDDING_DIM)

            self.MAX_SEQUENCE_LENGTH = max_sequence_length if (max_sequence_length > 0) else 1000
            print('Input/padding:\t\t=> ', self.MAX_SEQUENCE_LENGTH)
            
            self.batch_input_shape=(batch_size, self.MAX_SEQUENCE_LENGTH)

            self.MAX_NUM_WORDS = vocab_size if (vocab_size > 0) else 20000
            print('Vocab size:\t\t=> ', self.MAX_NUM_WORDS)

            self.max_cardinality = max_cardinality
            print('Max nodes cardinality:\t=> ', self.max_cardinality)

            self.tokenizer = tokenizer
            print('Tokenizer: \t\t=>  reloaded')

            self.show_response = show_feedback

            print('~~~~~~ Collect Embedding Indices ~~~~~~')
            self.embedding_indices = self.LoadGloVeEmbeddingIndices()
            print('Loaded word vectors:\t=> ', len(self.embedding_indices))

        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceDatasetsNodeValuesByEmbedding(self, datasets_nodes_values:list):
        """
        This function returns  embedding numpy arrays each dataset with stringified node values.
        The word embedding is directly collected from embedding_indices dictionairy.
        Remind, unknown words will be set to a vector of given embedding length with random values.
        Additionally, if you have different graph node cardinalities in your dataset, this tool gonna extend them to an equal size depending on the given max_cardinality.
            :param datasets_nodes_values:list: all datasets nodes values from GloVePreprocessor

            :returns: np.ndarray 3D
        """   
        try:
            datasets_nodes_initial_features = []
            max_dim = -1
            for dataset in datasets_nodes_values:
                dataset_nodes_initial_features = []
                for word in dataset:
                    word_embedding = self.embedding_indices.get(word)

                    if (word_embedding is None):  
                        word_embedding = np.random.randn(self.EMBEDDING_DIM)
                    
                    dataset_nodes_initial_features.append(word_embedding)

                assert (self.max_cardinality >= len(dataset_nodes_initial_features)), "ERROR: [Features Expansion FAILED], the given max cardinality was lower then the dataset max_cardinality!"
                datasets_nodes_initial_features.append(np.array(self.Extend2DByDim(dataset_nodes_initial_features)))

            return np.array(datasets_nodes_initial_features)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ReplaceDatasetsNodeValuesByEmbedding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Extend2DByDim(self, current_features:list):
        """
        This function expands a features list with 0 vectors until the defined max_cardinality is reached.
            :param current_features:list: given features list
        """   
        try:
            if self.max_cardinality > len(current_features):
                diff = self.max_cardinality - len(current_features)
                for _ in range(diff): current_features.append(np.zeros(self.EMBEDDING_DIM))

            assert (self.max_cardinality == len(current_features)), "ERROR: [Current_Size_Match FAILED]"
            return current_features
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.Extend2DByDim]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildVocabEmbeddingMatrix(self):
        """
        This function creates a weight matrix for all words in the vocab.
        Note, that words not found in embedding index, will be zeros.
        """   
        try:
            self.NUMBER_WORDS = min(self.MAX_NUM_WORDS, len(self.tokenizer.word_index)) + 1
            embedding_matrix = zeros((self.NUMBER_WORDS, self.EMBEDDING_DIM))
            for word, i in self.tokenizer.word_index.items():
                if i > self.MAX_NUM_WORDS: continue
                embedding_vector = self.embedding_indices.get(word)

                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
                else:
                    embedding_matrix[i] = np.random.randn(self.EMBEDDING_DIM)
            return embedding_matrix
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.BuildVocabEmbeddingMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LoadGloVeEmbeddingIndices(self):
        """
        This function load the word vector embedding indices from defined glove file.
        """   
        try:
            embeddings_index = dict()
            with open(self.GLOVE_DIR, 'r', encoding=self.GLOVE_ENC) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = asarray(values[1:], dtype=self.GLOVE_DTYPE)
                    embeddings_index[word] = coefs
            return embeddings_index
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.LoadGloVeEmbeddingIndices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildVocabEmbeddingLayer(self, embedding_matrix:np.ndarray):
        """
        This function load pre-trained word embeddings into an Embedding layer.
        Note that the embedding is set to not trainable to keep the embeddings fixed.
            :param embedding_matrix:np.ndarray: the vocab embedding matrix
        """   
        try:
            return Embedding(input_dim=self.NUMBER_WORDS,
                             output_dim=self.EMBEDDING_DIM,
                             batch_input_shape=self.batch_input_shape,
                             embeddings_initializer=Constant(embedding_matrix),
                             input_length=self.MAX_SEQUENCE_LENGTH,
                             trainable=False,
                             name=('glove_'+str(self.EMBEDDING_DIM)+'d_embedding'))
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.BuildVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ClearTokenizer(self):
        """
        Free the resource of the tokenizer cause they are not necessary later.
        """   
        try:
            self.tokenizer = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ClearTokenizer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ClearEmbeddingIndices(self):
        """
        Free the resource of the embedding indices cause they are not necessary later.
        """   
        try:
            self.embedding_indices = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ClearEmbeddingIndices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildGloveVocabEmbeddingLayer(self):
        """
        This function build the GloVe embedding layer.
        """   
        try:
            print('~~~~~~~~ Build Embedding Layer ~~~~~~~~')
            embedding_matrix = self.BuildVocabEmbeddingMatrix()
            if self.show_response: print('Embedding matrix:\n\t=> ',type(embedding_matrix))

            embedding_layer = self.BuildVocabEmbeddingLayer(embedding_matrix)
            if self.show_response: print('Embedding layer:\n\t=> ',type(embedding_layer))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            return embedding_layer
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.BuildGloveVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)