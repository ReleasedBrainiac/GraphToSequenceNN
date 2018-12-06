'''
    This part of my work is inspired by the code of:
    1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
    2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

    The GloVe dataset was provided at https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
'''
from numpy import asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.layers import Embedding
from DatasetHandler.ContentSupport import isNotNone, isIterable, isStr, isInt

class GloVeEmbedding:

    GLOVE_DIR = None
    GLOVE_DTYPE = 'float32'
    GLOVE_ENC = 'utf8'


    MAX_SEQUENCE_LENGTH = -1
    MAX_NUM_WORDS = -1
    EMBEDDING_DIM = -1
    VALIDATION_SPLIT = 0.2

    word_set = None
    tokenizer = None
    word_index = None
    number_words = -1
    
    max_length = -1

    # DONE FUNCTION
    def __init__(self, nodes_context, vocab_size=20000, max_sequence_length=1000, glove_file_path = '../../Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt', output_dim=100):
        """
        This class constructor stores the passed input sentences and the given GloVe embedding file path.
        Additionally it initialzes the max_length and the Keras tokenizer.
        The output_dim values should be adapted from the correpsonding pre-trained word vectors.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param nodes_context: the nodes context values of the dataset 
            :param vocab_size: maximum number of words to keep, based on word frequency
            :param max_sequence_length: max length length over all sequences (padding)
            :param glove_file_path: path of the desired GloVe word vector file
            :param output_dim: the general vector size for each word embedding
        """   
        try:
            if isStr(glove_file_path): self.GLOVE_DIR = glove_file_path

            if isInt(output_dim) and (output_dim > 0): self.EMBEDDING_DIM = output_dim

            if isInt(max_sequence_length) and (max_sequence_length > 0): self.MAX_SEQUENCE_LENGTH = max_sequence_length

            if isInt(vocab_size) and (vocab_size > 0): 
                self.MAX_NUM_WORDS = vocab_size
                self.tokenizer = Tokenizer(num_words=vocab_size, split=' ', char_level=False)                                            

            if isNotNone(nodes_context) and isIterable(nodes_context):
                print('Dataset Node Collection Started!')
                self.CollectWordSet(nodes_context)
                self.max_length = len(nodes_context)

        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

 
    def CollectWordSet(self, datasets):
        """
        This function collect all node values of each dataset into a set of unique values.
            :param datasets: list of amr parser datasets
        """   
        try:
            self.word_set = set()
            for dataset in datasets:
                node_dict = dataset[1][1]
                for key in node_dict:
                    self.word_set.add(node_dict[key])
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.CollectWordSet]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LoadGloVeEmbeddingIndices(self):
        """
        This function load the world vector embedding indices from defined glove file.
        Additionally, the function can load all whole vector embedding indices or only for the the unique context values, 
        depending on the switch value set at the constructor.
        """   
        try:
            embeddings_index = dict()
            print('File path: ', self.GLOVE_DIR)
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

    def TokenizeDatasets(self):
        """
        This function tokenize all text samples.
        """   
        try:
            print('Step_0')
            self.tokenizer.fit_on_texts(self.word_set)
            print('Step_1')
            sequences = self.tokenizer.texts_to_sequences(self.word_set)
            print('Step_2')
            self.word_index = self.tokenizer.word_index
            print('Length sequences ', len(sequences))
            return sequences
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.TokenizeDatasets]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def VectorizeDatasets(self, tokenized_sequences):
        """
        This function vectorize all tokenized text samples.
            :param tokenized_sequences: tokenized dataset samples
        """   
        try:
            return pad_sequences(tokenized_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.VectorizeDatasets]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
   
    def PrepareEmbeddingMatrix(self, embeddings_indexes):
        """
        This function creates a weight matrix for all words in training node context values.
        Remind, that words not found in embedding index, will be zeros.
            :param embeddings_indexes: indexes from GloVe loader
        """   
        try:
            self.number_words = min(self.MAX_NUM_WORDS, len(self.tokenizer.word_index)) + 1
            embedding_matrix = zeros((self.number_words, self.EMBEDDING_DIM))
            for word, i in self.tokenizer.word_index.items():
                if i > self.MAX_NUM_WORDS: continue
                embedding_vector = embeddings_indexes.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            return embedding_matrix
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.PrepareEmbeddingMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildEmbeddingLayer(self, embedding_matrix):
        """
        This function load pre-trained word embeddings into an Embedding layer.
        Note that trainable si set to False to keep the embeddings fixed.
            :param embedding_matrix: the embedding matrix collected from GloVe tokenization.
        """   
        try:
            return Embedding(self.number_words,
                             self.EMBEDDING_DIM,
                             embeddings_initializer=Constant(embedding_matrix),
                             input_length=self.MAX_SEQUENCE_LENGTH,
                             trainable=False)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.BuildEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetGloveEmbeddingLayer(self):
        """
        This function build and return the GloVe word to vector mapping layer.
        """   
        try:
            print('Dataset Vectorization Started!')
            embeddings_indexes = self.LoadGloVeEmbeddingIndices()
            print('Loaded %s word vectors.' % len(embeddings_indexes))

            tokenized_sequences = self.TokenizeDatasets()
            vectorized_sequences = self.VectorizeDatasets(tokenized_sequences)
            print('Found %s unique tokens.' % self.word_index)
            print('Shape of data tensor:', vectorized_sequences.shape)

            print('Embedding Process Started!')
            embedding_matrix = self.PrepareEmbeddingMatrix(embeddings_indexes)
            return self.BuildEmbeddingLayer(embedding_matrix)
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetParser.GetGloveEmbeddingLayer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)