'''
    This part of my work is inspired by the code of:
    1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
    2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    3. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

    The GloVe dataset was provided at https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
'''
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from DatasetHandler.ContentSupport import isNotNone, isIterable, isStr, isInt, isBool

#//TODO Tokenizer muss überarbeitet werden!

class GloVeEmbedding:

    GLOVE_DIR = None
    DATASET_DTYPE = 'float32'

    MAX_SEQUENCE_LENGTH = -1
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2


    context_values = None
    max_length = -1
    tokenizer = None
    
    vocab_size = -1
    
    out_dim = -1
    unique_words = None
    use_whole_glove = False


    def __init__(self, nodes_context, vocab_size, glove_file_path = '../../Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt', output_dim=100, use_whole_glove_ww=False):
        """
        This class constructor stores the passed input sentences and the given GloVe embedding file path.
        Additionally it initialzes the max_length and the Keras tokenizer.
        The output_dim values should be adapted from the correpsonding pre-trained word vectors.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param nodes_context: the nodes context values of the dataset 
            :param vocab_size: maximum number of words to keep, based on word frequency
            :param glove_file_path: path of the desired GloVe word vector file
            :param output_dim: the general vector size for each word embedding
            :param use_whole_glove_ww: switch allow to load the whole glove word vector or only the values for the unique values
        """   
        try:
            if isStr(glove_file_path): self.GLOVE_DIR = glove_file_path

            if isInt(output_dim) and (output_dim > 0): self.out_dim = output_dim

            if isBool(use_whole_glove_ww): self.use_whole_glove = use_whole_glove_ww

            if isInt(vocab_size) and (vocab_size > 0): 
                self.MAX_NUM_WORDS = vocab_size
                self.tokenizer = Tokenizer(num_words=vocab_size, lower=True, split=' ', char_level=False)                                            

            if isNotNone(nodes_context) and isIterable(nodes_context): 
                self.context_values = []
                for sentence in nodes_context: 
                    self.context_values.append(sentence)

                self.max_length = len(self.context_values)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


    def EncodedDocuments(self):
        """
        This function tokenize the given nodes context values and return the encoded context values. 
        The vocab size will be set additionally.
        """   
        try:
            self.tokenizer.fit_on_texts(self.context_values)
            sequences = self.tokenizer.texts_to_sequences(self.context_values)

            word_index = self.tokenizer.word_index
            
            return encoded_docs
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.EncodedDocuments]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    def SequencePadding(self,max_length_padding = 4, encoded_nodes_context_values=None):
        """
        This function padd all encoded nodes context values in a given representation with a defined range.
            :param max_length_padding: defines the padding range
            :param encoded_nodes_context_values: the name is enough
        """   
        try:
            return pad_sequences(encoded_nodes_context_values, maxlen=max_length_padding, padding='post')
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.SequencePadding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    # DONE FUNCTION
    def LoadGloVeEmbeddingIndices(self):
        """
        This function load the world vector embedding indices from defined glove file.
        Additionally, the function can load all whole vector embedding indices or only for the the unique context values, 
        depending on the switch value set at the constructor.
        """   
        try:
            embeddings_index = dict()
            with open(self.GLOVE_DIR) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = asarray(values[1:], dtype=self.DATASET_DTYPE)
                    embeddings_index[word] = coefs
            return embeddings_index
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.LoadGloVeEmbeddingIndices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    def GenerateTrainMatrix(self, embeddings_indexes):
        """
        This function creates a weight matrix for all words in training node context values
            :param embeddings_indexes: 
        """   
        try:
            embedding_matrix = zeros((self.vocab_size, self.out_dim))
            for word, i in self.tokenizer.word_index.items():
                embedding_vector = embeddings_indexes.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            return embedding_matrix
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.GenerateTrainMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetGloveEmbeddingLayer(self):
        """
        This function build and return the GloVe word to vector mapping layer.
        """   
        try:
            print('File path: ', self.GLOVE_DIR)
            embeddings_index = self.LoadGloVeEmbeddingIndices()
            print('Loaded %s word vectors.' % len(embeddings_index))

            #encoded_context_values = self.EncodedDocuments()
            #print('Encoded:', encoded_context_values)

            #padded_sequence = self.SequencePadding(max_length_padding=2,encoded_nodes_context_values=encoded_context_values)
            #print('Padded: ',padded_sequence)

            

            #embedding_matrix = self.GenerateTrainMatrix(embeddings_index)

            #return Embedding(self.vocab_size, self.out_dim, weights=[embedding_matrix], input_length=4, trainable=False)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.GetGloveEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)