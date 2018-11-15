# - *- coding: utf-8*-
'''
    Used Resources:
        => https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
        => https://www.pythonsheets.com/notes/python-rexp.html
'''

import re
from DatasetHandler.ContentSupport import isList, isStr, isInStr, isInt, isBool, isNone, isNotNone, setOrDefault
from DatasetHandler.FileReader import Reader
from DatasetHandler.FileWriter import Writer
from DatasetHandler.ExtractContentFromDataset import Extractor
from DatasetHandler.TextEvaluation import EvaluationHelpers
from Configurable.ProjectConstants import Constants
from TreeHandler.TreeParser import TParser
from TreeHandler.AMRGraphParser import GParser
from GraphHandler.GraphBuilder import GraphBuilder
from AMRHandler.AMRCleaner import Cleaner


class DatasetPipelines:

    # Variables inits
    look_up_extension_replace_path = './Datasets/LookUpAMR/supported_amr_internal_nodes_lookup.txt'
    extension_dict =  Reader(input_path=look_up_extension_replace_path).LineReadContent()

    # Class inits
    extractor = Extractor()
    constants = Constants()
    eval_Helper = EvaluationHelpers()
    gt_converter = GraphBuilder()
    t_parser = TParser()
    g_parser = None
    dataset_drop_outs = 0

    # Path content
    in_path = None
    out_path_extender = 'ouput'
    context_max_length = -1

    # Switches
    is_keeping_edges = False
    is_saving = False
    is_showing_feedback = False
    as_amr = False

    def __init__(self, in_path=None, output_path_extender=None, max_length=-1, saving=False,show_feedback=False, keep_edges=False):
        """
        This class constructor allow to set all path definbition for the dataset and the output. 
        Further its possible to define a maximal lengt for the used dataset. 
        Therefore the max_length define the exact value for the string length and doubled it for the  semantic string length.
        If it is negative the module gonna use all dataset elements.
        Additionally switches allow to tell the system:
            -> to save the content only
            -> to show feedback on the processing stage
            -> to include amr graph edge encoding
            :param in_path: dataset input path
            :param output_path_extender: result output path
            :param max_length: context length restriction
            :param saving: only saving flag
            :param show_feedback: show process content
            :param keep_edges: include edges in the amr cleaner strategy
        """   
        try:
            self.in_path = setOrDefault(in_path, self.constants.TYP_ERROR, isStr(in_path))
            self.dataset_drop_outs = 0

            if isStr(output_path_extender): self.out_path_extender = output_path_extender

            self.context_max_length = setOrDefault(max_length, -1, isInt(max_length))

            if isBool(show_feedback): self.is_showing_feedback = show_feedback

            if isBool(saving): self.is_saving = saving

            if isBool(keep_edges): self.is_keeping_edges = keep_edges

        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.__init__]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RemoveEnclosingAngleBracket(self, in_sentence):
        """
        This method clean up the a given amr extracted sentence from text formating markup.
            :param in_sentence: raw sentence AMR split element 
        """
        try:
            in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
            in_sentence = re.sub('</[^>]+>','', in_sentence)
            return  re.sub('<[^/>]+/>','', '#'+in_sentence)+'\n'
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.RemoveEnclosingAngleBracket]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeAmrSemanticString(self, semantic, semantic_flag):
        """
        This function allows to convert a raw AMR semantic into a cleaned and minimalized version of it.
            :param semantic: raw semantic input
            :param semantic_flag: marker/delim added to the cleaned semantic
        """
        try:
            cleaner = Cleaner(input_context=semantic, input_extension_dict=self.extension_dict, keep_edges=self.is_keeping_edges)
            if cleaner.isCleaned:
                self.extension_dict = cleaner.extension_dict
                return '#'+semantic_flag+' \n'+cleaner.cleaned_context+'\n'+'\n'
            else:
                self.dataset_drop_outs += 1
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrSemanticString]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeAmrTree(self, semantic, semantic_flag):
        """
        This function allows to convert a raw AMR semantic (graph-) string
        into a cleaned and minimalized anytree reprenstation of it.
            :param semantic: raw semantic input
            :param semantic_flag: marker/delim to add to cleaned semantic
        """
        try:
            out = '#'+semantic_flag+' '+semantic+'\n'+'\n'
            return self.t_parser.Execute(out, semantic_flag, self.is_showing_feedback, self.is_saving)
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrTree]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasetPair(self, sentence_flag, semantic_flag, sentence, semantic):
        """
        This function collect a data pair from raw sentences and semantics.
        ATTENTION: [as_amr = true] does not support conversion with ConvertToTensorMatrices!
            :param sentence_flag: a marker to attach to the cleaned sentence, make it easier to find later
            :param semantic_flag: a marker to attach to the cleaned semantic, make it easier to find later
            :param sentence: raw input of the AMR sentence
            :param semantic: raw input of the AMR semantic
        """
        try:
            sentence = sentence_flag+' '+sentence
            sentence = self.RemoveEnclosingAngleBracket(sentence)
            if(self.as_amr):
                semantic = self.ForgeAmrSemanticString(semantic, semantic_flag)
            else: 
                semantic = self.ForgeAmrTree(semantic, semantic_flag)
                
            if isNotNone(semantic): 
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectDatasetPair]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectAllDatasetPairs(self, sentence_flag, semantic_flag, sent_array, sem_array):
        """
        This function collect multiples pairs of semantic and sentence data as list of data pairs.
        For this case we pass arrays of raw sentences and semantics, 
        where index i in both arrays point to a sentence and the corresponding semantic.
            :param sentence_flag: a marker to attach to the cleaned sentence, make it easier to find later
            :param semantic_flag: a marker to attach to the cleaned semantic, make it easier to find later
            :param sent_array: Array of the raw input of the AMR sentences
            :param sem_array: Array of the raw input of the AMR semantics
        """
        try:
            dataset_pairs_sent_sem = []
            for i in range(min(len(sent_array), len(sem_array))):
                dataset_pairs_sent_sem.append(self.CollectDatasetPair(sentence_flag, semantic_flag, sent_array[i], sem_array[i]))

            return dataset_pairs_sent_sem
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectAllDatasetPairs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Pipeline(self):
        """
        This function collect the cleaned sentence graphs as:
        1. AMR string representation [as_amr_structure = True]
        2. AnyTree [else]:
            1. as JSON     [is_not_saving = true]
            2. as AnyNode  [else]
        """
        try:

            dataset = Reader(self.in_path).GroupReadAMR()
            dataset=dataset[1:len(dataset)]

            extracted = self.extractor.Extract(in_content=dataset, 
                                               max_len=self.context_max_length, 
                                               x_delim=self.constants.SENTENCE_DELIM, 
                                               y_delim=self.constants.FILE_DELIM)        

            mean_value_sentences = self.eval_Helper.CalculateMeanValue(extracted[0])
            mean_value_semantics = self.eval_Helper.CalculateMeanValue(extracted[1])     

            data_pairs = self.CollectAllDatasetPairs(self.constants.SENTENCE_DELIM, 
                                                     self.constants.SEMANTIC_DELIM, 
                                                     extracted[2], 
                                                     extracted[3])

            print('Max restriction: ', self.context_max_length)
            print('Path input: ', self.in_path)
            print('Count sentences: ', len(extracted[2]))
            print('Count semantics: ', len(extracted[3]))
            print('Mean sentences: ', mean_value_sentences)
            print('Mean semantics: ', mean_value_semantics)
            print('Data dropouts:', self.dataset_drop_outs)

            return data_pairs
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SaveData(self, as_amr):
        """
        This function calls the Pipeline and store the desired results.
            :param as_amr: if True save as amr string otherwise as anytree json string
        """
        try:
            
            if isBool(as_amr): self.as_amr = as_amr
            writer = Writer(self.in_path, 
                            self.out_path_extender, 
                            self.Pipeline())
            
            return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.SaveData]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    def ProvideData(self):
        """
        This function calls the Pipeline and return the cleaned dataset for ANN usage.
        """
        try:
            return self.gt_converter.GetDataSet(self.Pipeline())
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ProvideData]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)