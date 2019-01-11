'''
    This part of my work is inspired by the code of:
    1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
    2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    3. https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings 
    4. https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html 

    The GloVe dataset was provided at https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
'''
import sys
import numpy as np
from numpy import asarray, zeros
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from DatasetHandler.ContentSupport import isStr, isInt, isBool, isDict, isNotNone


class GloVeEmbedding:

    GLOVE_DIR = None
    GLOVE_DTYPE = 'float32'
    GLOVE_ENC = 'utf8'

    MAX_SEQUENCE_LENGTH = -1
    MAX_NUM_WORDS = -1
    EMBEDDING_DIM = -1

    show_response = False
    number_words = -1
    final_tokenizer = None
    embedding_indices = None


    def __init__(self, tokenizer, vocab_size=20000, max_sequence_length=1000, glove_file_path = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt', output_dim=100, show_feedback=False):
        """
        This class constructor stores all given parameters. 
        Further it execute the word collecting process from the datasets node dictionairies.
        The output_dim values should be adapted from the correpsonding pre-trained word vectors.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param tokenizer: tokenizer from GloVe dataset preprocessing.
            :param vocab_size: maximum number of words to keep, based on word frequency
            :param max_sequence_length: max length length over all sequences (padding)
            :param glove_file_path: path of the desired GloVe word vector file
            :param output_dim: the general vector size for each word embedding
            :param show_feedback: switch allows to show process response on console or not
        """   
        try:
            print('######## Init Embedding GloVe #########')
            if isStr(glove_file_path): 
                self.GLOVE_DIR = glove_file_path
                print('GloVe file:\t\t', self.GLOVE_DIR)

            if isInt(output_dim) and (output_dim > 0): 
                self.EMBEDDING_DIM = output_dim
                print('Output dimension:\t', self.EMBEDDING_DIM)

            if isInt(max_sequence_length) and (max_sequence_length > 0): 
                self.MAX_SEQUENCE_LENGTH = max_sequence_length
                print('Input/padding:\t\t', self.MAX_SEQUENCE_LENGTH)

            if isInt(vocab_size) and (vocab_size > 0): 
                self.MAX_NUM_WORDS = vocab_size
                print('Vocab size:\t\t', self.MAX_NUM_WORDS)

            if isNotNone(tokenizer): 
                self.final_tokenizer = tokenizer
                print('Tokenizer: \t\t reloaded')

            if isBool(show_feedback): self.show_response = show_feedback

            print('###### Collect Embedding Indices ######')
            self.embedding_indices = self.LoadGloVeEmbeddingIndices()
            if self.show_response: print('\t=> Loaded %s word vectors.' % len(self.embedding_indices))

        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceDatasetsNodeValuesByEmbedding(self, datasets_nodes_values):
        try:
            datasets_nodes_initial_features = []

            for dataset in datasets_nodes_values:
                dataset_nodes_initial_features = []
                for word in dataset: 
                    word_embedding = self.embedding_indices.get(word)
                    dataset_nodes_initial_features.append(word_embedding)
                
                datasets_nodes_initial_features.append(dataset_nodes_initial_features)
                if len(dataset) != len(dataset_nodes_initial_features): 
                    print('ERROR: [Current_Size_Match FAILED]')
                    sys.exit(0)
                
            if len(datasets_nodes_values) != len(datasets_nodes_initial_features): 
                print('ERROR: [Size_Match FAILED]')
                sys.exit(0)

            return datasets_nodes_initial_features
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ReplaceDatasetNodeValuesByEmbedding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildVocabEmbeddingMatrix(self, embedding_indices):
        """
        This function creates a weight matrix for all words in the vocab.
        Note, that words not found in embedding index, will be zeros.
            :param embedding_indices: the indices from GloVe loader
        """   
        try:
            if self.show_response: print('Building vocab embedding matrix!')
            self.number_words = min(self.MAX_NUM_WORDS, len(self.final_tokenizer.word_index)) + 1
            embedding_matrix = zeros((self.number_words, self.EMBEDDING_DIM))
            for word, i in self.final_tokenizer.word_index.items():
                if i > self.MAX_NUM_WORDS: continue
                embedding_vector = embedding_indices.get(word)

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
            if self.show_response: print('Loading GloVe indices!')
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

    def BuildVocabEmbeddingLayer(self, embedding_matrix):
        """
        This function load pre-trained word embeddings into an Embedding layer.
        Note that the embedding is set to not trainable to keep the embeddings fixed.
            :param embedding_matrix: the vocab embedding matrix
        """   
        try:
            if self.show_response: print('Building vocab embedding layer!')
            return Embedding(self.number_words,
                             self.EMBEDDING_DIM,
                             embeddings_initializer=Constant(embedding_matrix),
                             input_length=self.MAX_SEQUENCE_LENGTH,
                             trainable=False)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.BuildVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ClearTokenizer(self):
        """
        Free the resource of the tokenizer cause they are not necessary later.
            :param self: 
        """   
        try:
            self.final_tokenizer = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ClearTokenizer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ClearEmbeddingIndices(self):
        """
        Free the resource of the embedding indices cause they are not necessary later.
            :param self: 
        """   
        try:
            self.final_tokenizer = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.ClearEmbeddingIndices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildGloveVocabEmbeddingLayer(self):
        """
        This function build the GloVe embedding layer.
        """   
        try:
            
            print('######## Build Embedding Layer ########')
            embedding_matrix = self.BuildVocabEmbeddingMatrix(self.embedding_indices)
            if self.show_response: print('\t=> Embedding matrix:\n',embedding_matrix,'.')

            embedding_layer = self.BuildVocabEmbeddingLayer(embedding_matrix)
            if self.show_response: print('\t=> Embedding layer:\n',embedding_layer,'.')

            self.ClearTokenizer()
            print('#######################################')
            return embedding_layer
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeEmbeddingLayer.BuildGloveVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)