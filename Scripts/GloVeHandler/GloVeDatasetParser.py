'''
    This part of my work is inspired by the code of:
    1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
    2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    3. https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings 

    The GloVe dataset was provided at https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
'''
from oset import oset
import numpy as np
from numpy import asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.layers import Embedding
from DatasetHandler.ContentSupport import isNotNone, isIterable, isStr, isInt, isBool, isODict, isList

class GloVeEmbedding:

    GLOVE_DIR = None
    GLOVE_DTYPE = 'float32'
    GLOVE_ENC = 'utf8'

    MAX_SEQUENCE_LENGTH = -1
    MAX_NUM_WORDS = -1
    EMBEDDING_DIM = -1

    tokenizer = None
    show_response = False

    word_set = None
    edge_set = None
    word_index = None
    number_words = -1
    max_length = -1

    def __init__(self, nodes_context, vocab_size=20000, max_sequence_length=1000, glove_file_path = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt', output_dim=100, show_feedback=False):
        """
        This class constructor stores all given parameters. 
        Further it execute the word collecting process from the datasets node dictionairies.
        The output_dim values should be adapted from the correpsonding pre-trained word vectors.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param nodes_context: the nodes context values of the dataset 
            :param vocab_size: maximum number of words to keep, based on word frequency
            :param max_sequence_length: max length length over all sequences (padding)
            :param glove_file_path: path of the desired GloVe word vector file
            :param output_dim: the general vector size for each word embedding
            :param show_feedback: switch allows to show process response on console or not
        """   
        try:
            print('################ Setup ################')
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
                self.tokenizer = Tokenizer(num_words=vocab_size, split=' ', char_level=False)
                print('Vocab size:\t\t', self.MAX_NUM_WORDS)

            if isBool(show_feedback): self.show_response = show_feedback
            print('#######################################')

            if isNotNone(nodes_context) and isIterable(nodes_context): self.CollectVocab(nodes_context)

        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectValues(self, node_values, edge_values):

        if isODict(node_values): node_values = node_values.values()


        for idx, elem in enumerate(node_values):
            if isNotNone(edge_values) and isNotNone(node_values[elem]):
                if isODict(node_values) and isNotNone(node_values[elem]):
                    self.word_set.add(node_values[elem])
                    self.edge_set.add(edge_values[idx])
                elif isList(node_values) and isNotNone(elem): 
                    self.word_set.add(elem)
                    self.edge_set.add(edge_values[idx])
                else:
                    print('Found None: ', elem , ' | ', node_values[elem])

    def CollectVocab(self, datasets):
        """
        This function collect all node values from each dataset into a set of unique values (vocab).
            :param datasets: list of cleaned and parsed amr datasets
        """   
        try:
            if self.show_response: print('Collecting vocab!')
            self.word_set = oset()
            self.edge_set = oset()
            for dataset in datasets:
                node_values = dataset[1][1]
                edge_values = dataset[1][0]
                print('length control: ', (len(node_values) == len(edge_values)))

                if len(node_values) == len(edge_values): self.CollectValues(node_values, edge_values)

        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.CollectVocab]. Arguments:\n{1!r}"
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
            template = "An exception of type {0} occurred in [GloVeDatasetParser.LoadGloVeEmbeddingIndices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def TokenizeVocab(self):
        """
        This function tokenize the collected vocab and set the global word index list.
        """   
        try:
            if self.show_response: print('Tokenize vocab!')
            self.tokenizer.fit_on_texts(self.word_set)
            sequences = self.tokenizer.texts_to_sequences(self.word_set)
            self.word_index = self.tokenizer.word_index
            return sequences
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.TokenizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def VectorizeVocab(self, tokenized_sequences):
        """
        This function vectorize all tokenized vocab samples.
            :param tokenized_sequences: tokenized vocab samples
        """   
        try:
            if self.show_response: print('Vectorize vocab!')
            return pad_sequences(tokenized_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.VectorizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
   
    def BuildVocabEmbeddingMatrix(self, embeddings_indexes):
        """
        This function creates a weight matrix for all words in the vocab.
        Note, that words not found in embedding index, will be zeros.
            :param embeddings_indexes: the indices from GloVe loader
        """   
        try:
            if self.show_response: print('Building vocab embedding matrix!')
            self.number_words = min(self.MAX_NUM_WORDS, len(self.tokenizer.word_index)) + 1
            embedding_matrix = zeros((self.number_words, self.EMBEDDING_DIM))
            for word, i in self.tokenizer.word_index.items():
                if i > self.MAX_NUM_WORDS: continue
                embedding_vector = embeddings_indexes.get(word)

                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
                else:
                    embedding_matrix[i] = np.random.randn(self.EMBEDDING_DIM)
            return embedding_matrix
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.BuildVocabEmbeddingMatrix]. Arguments:\n{1!r}"
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
            template = "An exception of type {0} occurred in [GloVeDatasetParser.BuildVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildGloveVocabEmbeddingLayer(self):
        """
        This function build and return the GloVe word to vector mapping layer, depending on the vocab.
        """   
        try:
            embeddings_indexes = self.LoadGloVeEmbeddingIndices()
            if self.show_response: print('\t=> Loaded %s word vectors.' % len(embeddings_indexes))

            tokenized_sequences = self.TokenizeVocab()
            if self.show_response: print('\t=> Found %s unique tokens.' % len(self.word_index))

            vectorized_sequences = self.VectorizeVocab(tokenized_sequences)
            if self.show_response: print('\t=> Fixed',vectorized_sequences.shape,'data tensor.')

            embedding_matrix = self.BuildVocabEmbeddingMatrix(embeddings_indexes)
            if self.show_response: print('\t=> Embedding matrix:\n',embedding_matrix,'.')

            return self.BuildVocabEmbeddingLayer(embedding_matrix)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.BuildGloveVocabEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)