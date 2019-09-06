import collections
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from DatasetHandler.ContentSupport import isNotNone
from Plotter.PlotBarChart import BarChart

class GloVeDatasetPreprocessor:
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

            self._node_words_list:list = None
            self._edge_matrices_fw:list = None
            self._edge_matrices_bw:list = None

            self._min:int = -1
            self._max:int = -1
            self._word_to_none_ratios:dict = {}

            self._sentences_dec_in:list = None
            self._sentences_tar_in:list = None
            self._word_index = None
            self._tokenizer_words = None
            self._show_response = show_feedback               

            self.MAX_SEQUENCE_LENGTH = max_sequence_length  if (max_sequence_length > 0) else 1000
            self._tokenizer = None if (vocab_size < 1) else Tokenizer(num_words=vocab_size, split=' ', char_level=False, filters='')

            print('Input/padding:\t\t => ', self.MAX_SEQUENCE_LENGTH)
            print('Vocab size:\t\t => ', vocab_size)
            
            if isNotNone(nodes_context): 
                self.CollectDatasamples(nodes_context)
                if self._show_response: 
                    print('Collected decoder samples:\t => ', len(self._sentences_dec_in))
                    print('Generated target samples:\t => ', len(self._sentences_tar_in)) 

                
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
            sentence = sentence.replace('#::snt ', '')
            sentence = sentence.replace('- -','-')
            sentence = sentence.replace('"', '')
            sentence = sentence.replace('(','')
            sentence = sentence.replace(')','')
            return sentence.replace('  ',' ')
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
            self._node_words_list = []
            self._edge_matrices_fw = []
            self._edge_matrices_bw = []
            self._sentences_dec_in = []
            self._sentences_tar_in = []
            self._word_to_none_ratios = {}

            for dataset in datasets: 
                if isNotNone(dataset[1][0]) and len(dataset[1][0]) == 2 and len(dataset[1][0][0]) == len(dataset[1][0][1]):
                    node_words = dataset[1][1]
                    self._node_words_list.append(node_words)

                    # Collect the none ratio values for bar chart
                    summed_nones = sum(x is not None for x in node_words)
                    self._min = min(self._min, summed_nones) 
                    self._max = min(self._max, summed_nones) 
                    self._word_to_none_ratios[len(node_words)] = summed_nones

                    self._edge_matrices_fw.append(dataset[1][0][0])
                    self._edge_matrices_bw.append(dataset[1][0][1])

                    input_t_0 = self.ReplaceSentenceFlagAndDialogElements(dataset[0])
                    input_t_minus_1 = input_t_0.split(' ', 1)[1]

                    self._sentences_dec_in.append(input_t_0)
                    self._sentences_tar_in.append(input_t_minus_1)
            assert (len(self._sentences_dec_in) == len(self._sentences_tar_in)), "Dataset Error! Inputs counter doesn't match targets counter"
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.CollectDatasamples]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def TokenizeVocab(self):
        """
        This function tokenizes the collected vocab (sentences, targets) and set the global word index list.
        """   
        try:
            if self._show_response: print('Tokenize vocab!')
            self._tokenizer.fit_on_texts(self._sentences_dec_in)
            sequences_in_dec = self._tokenizer.texts_to_sequences(self._sentences_dec_in)
            sequences_in_tar = self._tokenizer.texts_to_sequences(self._sentences_tar_in)
            self._word_index = self._tokenizer.word_index
            return sequences_in_dec, sequences_in_tar
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.TokenizeVocab]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def TokenizeNodes(self, node_lists:list):
        """
        This method convert nodes word lookup to nodes word index look up.
            :param node_lists:list: list of node lists
        """   
        try:
            look_up = {y:x for x,y in self._tokenizer.index_word.items()} 
            tokenized_nodes_all = []

            for nodes in node_lists:
                tokenized_nodes = []

                for node in nodes:
                    tokenized_nodes.append(look_up.get(node, 0))

                tokenized_nodes_all.append(tokenized_nodes)

            return tokenized_nodes_all
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.TokenizeNodes]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def VectorizeVocab(self, tokenized_sequences:list):
        """
        This function vectorizes all tokenized vocab samples.
            :param tokenized_sequences:list: tokenized vocab samples
        """   
        try:
            if self._show_response: print('Vectorize vocab!')
            padded_sequences = pad_sequences(tokenized_sequences, padding='post', maxlen=self.MAX_SEQUENCE_LENGTH)
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
            if self._show_response: print('\t=> Found %s unique tokens.' % len(self._word_index))

            vectorized_dec_ins, indices_dec = self.VectorizeVocab(tokenized_dec_ins)
            if self._show_response: 
                print('\t=> Fixed',vectorized_dec_ins.shape,'decoder input tensor.')

            vectorized_tar_ins, indices_tar = self.VectorizeVocab(tokenized_tar_ins)
            if self._show_response: 
                print('\t=> Fixed',vectorized_tar_ins.shape,'decoder target tensor.')

            assert (indices_dec.all() == indices_tar.all()), "Indices missmatch for vectorized decoder inputs and targets!"
            self._tokenizer_words = self._tokenizer.word_index.items()
            self._edge_matrices_fw = np.array(self._edge_matrices_fw)
            self._edge_matrices_bw = np.array(self._edge_matrices_bw)

            print('Result structure! \n\t=> [RawTextDecIn, RawTextTarIn, EdgesFw, EdgesBW, VectorizedDecIn, VectorizedTarIn, GraphNodesValuesList, VectorizedInputsIndices]')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            return [self._sentences_dec_in, self._sentences_tar_in, self._edge_matrices_fw, self._edge_matrices_bw, vectorized_dec_ins, vectorized_tar_ins, self._node_words_list, indices_dec]
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def FreeUnusedResources(self):
        """
        This function set sentences_dec_in and  sentences_tar_in to None.
        """
        try:
            self._sentences_dec_in = None
            self._sentences_tar_in = None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.FreeUnusedResources]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Convert(self, prefix:str, lang:Tokenizer, tensor):
        """
        Used from example => https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention to test my stuff.
            :param prefix:str: string prefix
            :param lang:Tokenizer: word embedding
            :param tensor: testable tensor
        """
        try:
            print("[", prefix, "]")
            for t in tensor:
                if t!=0: print ("%d ----> %s" % (t, lang.index_word[t]))
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.Convert]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def PlotNoneRatio(self, path:str = None):
        """
        This method plot the word values to none values ratio collect from the nodes lists.
            :param path:str: path combined with filename for file image storing
        """
        try:
            BarChart(   dataset = self._word_to_none_ratios, 
                        min = self._min,
                        max = self._max, 
                        title = 'None value occurences in graph nodes', 
                        short_title = 'Occurence min and max', 
                        x_label = 'Node values', 
                        y_label = 'Found None values',
                        path = path)
            return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [GloVeDatasetPreprocessor.PlotNoneRatio]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
                