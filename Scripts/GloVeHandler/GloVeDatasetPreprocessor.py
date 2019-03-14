import collections
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from DatasetHandler.ContentSupport import isNotNone

class GloVeDatasetPreprocessor():
    """
    This class preprocesses given Datapairs from DatasetHandler.
    This includes:
        1. creating a tokenizer
        2. collecting datapair samples in lists for the usage in neutal networks
        3. tokenizing and padding sentences
        4. returning them additionally to word and paddding indices

    This part of my work is partially inspired by the code of:
        1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
        2. https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
        3. https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings 
        4. https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html 
    """

    def __init__(self, nodes_context:list, vocab_size:int =20000, max_sequence_length:int =1000, show_feedback:bool =False):
        """
        This class constructor stores all given parameters. 
        Further it execute the word collecting process from the datasets node dictionairies.
        For further informations take a look at => https://nlp.stanford.edu/projects/glove/ => [Download pre-trained word vectors]
            :param nodes_context:list: the nodes context values of the dataset 
            :param vocab_size:int: maximum number of words to keep, based on word frequency
            :param max_sequence_length:int: max length length over all sequences (padding)
            :param show_feedback:bool: switch allows to show process response on console or not
        """   
        try:
            print('~~~~~~ GloVe Dataset Preprocessor ~~~~~')

            self.node_words_list = None
            self.edge_matrices_fw = None
            self.edge_matrices_bw = None

            self.sentences_dec_in = None
            self.sentences_tar_in = None
            self.word_index = None
            self.tokenizer_words = None
            self.show_response = show_feedback               

            self.MAX_SEQUENCE_LENGTH = max_sequence_length  if (max_sequence_length > 0) else 1000
            self.tokenizer = None if (vocab_size < 1) else Tokenizer(num_words=vocab_size, split=' ', char_level=False)

            print('Input/padding:\t\t => ', self.MAX_SEQUENCE_LENGTH)
            print('Vocab size:\t\t => ', vocab_size)
            
            if isNotNone(nodes_context): 
                self.CollectDatasamples(nodes_context)
                if self.show_response: 
                    print('Collected decoder samples:\t => ', len(self.sentences_dec_in))
                    print('Generated target samples:\t => ', len(self.sentences_tar_in)) 

                
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceSentenceFlagAndDialogElements(self, sentence:str):
        """
        This function removes the sentence flag and some dialog elements.
            :param sentence:str: input sentence
        """
        try:
            return sentence.replace('#::snt ', '').replace('" ', '').replace(' "', '').replace('- -','-')
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.ReplaceSentenceFlagAndDialogElements]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasamples(self, datasets:list):
        """
        This function collects all dataset word lists and edge matrices for further processing.
        Additionally it generates the lstm decoder targets from decoder inputs (depending on seq2seq encoder decoder)!
            :param datasets:list: pairs of cleaned amr datasets
        """   
        try:
            self.node_words_list = []
            self.edge_matrices_fw = []
            self.edge_matrices_bw = []
            self.sentences_dec_in = []
            self.sentences_tar_in = []

            for dataset in datasets: 
                if isNotNone(dataset[1][0]) and len(dataset[1][0]) == 2 and len(dataset[1][0][0]) == len(dataset[1][0][1]):
                    self.node_words_list.append(dataset[1][1])
                    self.edge_matrices_fw.append(dataset[1][0][0])
                    self.edge_matrices_bw.append(dataset[1][0][1])

                    ph_dec_in = self.ReplaceSentenceFlagAndDialogElements(dataset[0])
                    ph_tar_in = ph_dec_in.split(' ', 1)[1]
                    
                    self.sentences_dec_in.append(ph_dec_in)
                    self.sentences_tar_in.append(ph_tar_in)
            assert (len(self.sentences_dec_in) == len(self.sentences_tar_in)), "Dataset Error! Inputs counter doesn't match targets counter"
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.CollectDatasamples]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def TokenizeVocab(self):
        """
        This function tokenizes the collected vocab (sentences, targets) and set the global word index list.
        """   
        try:
            if self.show_response: print('Tokenize vocab!')
            self.tokenizer.fit_on_texts(self.sentences_dec_in)
            sequences_in_dec = self.tokenizer.texts_to_sequences(self.sentences_dec_in)
            sequences_in_tar = self.tokenizer.texts_to_sequences(self.sentences_tar_in)
            self.word_index = self.tokenizer.word_index
            return sequences_in_dec, sequences_in_tar
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.TokenizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def VectorizeVocab(self, tokenized_sequences:list):
        """
        This function vectorizes all tokenized vocab samples.
            :param tokenized_sequences:list: tokenized vocab samples
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

    def Execute(self):
        """
        This function returns all given data samples with tokenized sequences mapping for there nodes context.
        Structure: [sentences, edges, vectorized_dec_ins, nodes, indices]
        """   
        try:
            print('~~~~~~~~~~~~~ Prepare Data ~~~~~~~~~~~~')
            tokenized_dec_ins, tokenized_tar_ins = self.TokenizeVocab()
            if self.show_response: print('\t=> Found %s unique tokens.' % len(self.word_index))

            vectorized_dec_ins, indices_dec = self.VectorizeVocab(tokenized_dec_ins)
            vectorized_tar_ins, indices_tar = self.VectorizeVocab(tokenized_tar_ins)

            assert (indices_dec.all() == indices_tar.all()), "Indices missmatch for vectorized decoder inputs and targets!"

            if self.show_response: 
                print('\t=> Fixed',vectorized_dec_ins.shape,'decoder input tensor.')
                print('\t=> Fixed',vectorized_tar_ins.shape,'decoder target tensor.')

            self.tokenizer_words = self.tokenizer.word_index.items()

            print('Result structure! \n\t=> [RawTextDecIn, RawTextTarIn, EdgesFw, EdgesBW, VectorizedDecIn, VectorizedTarIn, GraphNodesValuesList, VectorizedInputsIndices]')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            return [self.sentences_dec_in, self.sentences_tar_in, self.edge_matrices_fw, self.edge_matrices_bw, vectorized_dec_ins, vectorized_tar_ins, self.node_words_list, indices_dec]
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def FreeUnusedResources(self):
        """
        This function set sentences_dec_in and  sentences_tar_in to None.
        """
        try:
            self.sentences_dec_in = None
            self.sentences_tar_in = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.FreeUnusedResources]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        