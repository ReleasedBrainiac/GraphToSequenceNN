'''
    This part of my work is inspired by the code of:
    1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
    2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    3. https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings 
    4. https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html 

    The GloVe dataset was provided at https://nlp.stanford.edu/projects/glove/#Download%20pre-trained%20word%20vectors 
'''
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from DatasetHandler.ContentSupport import isNotNone, isIterable, isStr, isInt, isBool

class GloVeDatasetPreprocessor:

    MAX_SEQUENCE_LENGTH = -1

    tokenizer = None
    show_response = False

    node_words_list = None
    edge_matrices = None
    sentences_list = None
    word_index = None
    tokenizer_words = None

    def __init__(self, nodes_context, vocab_size=20000, max_sequence_length=1000, show_feedback=False):
        """
        This class constructor stores all given parameters. 
        Further it execute the word collecting process from the datasets node dictionairies.
        The output_dim values should be adapted from the correpsonding pre-trained word vectors.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param nodes_context: the nodes context values of the dataset 
            :param vocab_size: maximum number of words to keep, based on word frequency
            :param max_sequence_length: max length length over all sequences (padding)
            :param show_feedback: switch allows to show process response on console or not
        """   
        try:
            print('~~~~~~ GloVe Dataset Preprocessor ~~~~~')
            if isBool(show_feedback): self.show_response = show_feedback

            if isInt(max_sequence_length) and (max_sequence_length > 0): 
                self.MAX_SEQUENCE_LENGTH = max_sequence_length
                print('Input/padding:\t\t => ', self.MAX_SEQUENCE_LENGTH)

            if isInt(vocab_size) and (vocab_size > 0): 
                self.tokenizer = Tokenizer(num_words=vocab_size, split=' ', char_level=False)
                print('Vocab size:\t\t => ', vocab_size)
            
            if isNotNone(nodes_context) and isIterable(nodes_context): 
                self.CollectDatasamples(nodes_context)
                
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceSentenceFlagAndDialogElements(self, sentence):
        """
        This function retrun a sentence without sentence flag and some direct speech elements.
            :param sentence: input sentence
        """
        try:
            return sentence.replace('#::snt ', '').replace('" ', '').replace(' "', '').replace('- -','-')
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.ReplaceSentenceFlagAndDialogElements]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasamples(self, datasets):
        """
        This function collect all dataset word lists and edge matrices for further processing.
            :param datasets: list of cleaned and parsed amr datasets
        """   
        try:
            self.node_words_list = []
            self.edge_matrices = []
            self.sentences_list = []

            for dataset in datasets: 
                if isNotNone(dataset[1][0]) and len(dataset[1][0]) == 2 and len(dataset[1][0][0]) == len(dataset[1][0][1]):
                    self.node_words_list.append(dataset[1][1])
                    self.edge_matrices.append(dataset[1][0])
                    self.sentences_list.append(self.ReplaceSentenceFlagAndDialogElements(dataset[0]))
            if self.show_response: print('Collected samples:\t => ', len(self.sentences_list))
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.CollectDatasamples]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def TokenizeVocab(self):
        """
        This function tokenize the collected vocab and set the global word index list.
        """   
        try:
            if self.show_response: print('Tokenize vocab!')
            self.tokenizer.fit_on_texts(self.sentences_list)
            sequences = self.tokenizer.texts_to_sequences(self.sentences_list)
            self.word_index = self.tokenizer.word_index
            return sequences
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.TokenizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def VectorizeVocab(self, tokenized_sequences):
        """
        This function vectorize all tokenized vocab samples.
            :param tokenized_sequences: tokenized vocab samples
        """   
        try:
            if self.show_response: print('Vectorize vocab!')
            padded_sequences = pad_sequences(tokenized_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
            indices = np.arange(padded_sequences.shape[0])
            return padded_sequences, indices
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.VectorizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)                

    def GetPreparedDataSamples(self):
        """
        This function return all given data samples with replaced GloVe word to vector mapping for there nodes context.
        Structure: [sentences, edges, vectorized_sequences, nodes, indices]
        """   
        try:
            print('~~~~~~~~~~~~~ Prepare Data ~~~~~~~~~~~~')
            tokenized_sequences = self.TokenizeVocab()
            if self.show_response: print('\t=> Found %s unique tokens.' % len(self.word_index))

            vectorized_sequences, indices = self.VectorizeVocab(tokenized_sequences)
            if self.show_response: print('\t=> Fixed',vectorized_sequences.shape,'data tensor.')

            self.tokenizer_words = self.tokenizer.word_index.items()

            print('Result structure! \n\t=> [Sentences, EdgeArrays, VectorizedSequencesLists, GraphNodesValuesList, SequenceIndices]')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            return [self.sentences_list, self.edge_matrices, vectorized_sequences, self.node_words_list, indices]
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.GetPreparedDataSamples]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)