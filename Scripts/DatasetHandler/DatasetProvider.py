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
from GraphHandler.GraphTreeConverter import GTConverter
from AMRHandler.AMRCleaner import Cleaner


class DatasetPipelines:

    # Variables inits
    parenthesis = ['(', ')']
    look_up_extension_replace_path = './Datasets/LookUpAMR/supported_amr_internal_nodes_lookup.txt'
    extension_dict =  Reader(input_path=look_up_extension_replace_path).LineReadContent()

    # Class inits
    extractor = Extractor()
    constants = Constants()
    eval_Helper = EvaluationHelpers()
    gt_converter = GTConverter()
    t_parser = TParser()
    g_parser = None

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
        try:
            self.in_path = setOrDefault(in_path, self.constants.TYP_ERROR, isStr(in_path))

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
            in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
            return in_sentence+'\n'
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.RemoveEnclosingAngleBracket]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CleanReforgeAmrSemantic(self, semantic, semantic_flag):
        """
        This function allows to convert a raw AMR semantic (graph-) string  
        into a cleaned and minimalized version of it.
            :param semantic: raw semantic input
            :param semantic_flag: marker/delim to add to cleaned semantic
        """
        try:
            cleaner = Cleaner(self.parenthesis, 
                              input_context=semantic,
                              input_extension_dict=self.extension_dict,
                              keep_edges = self.is_keeping_edges
                              )

            self.extension_dict = cleaner.extension_dict
            half_cleaned_sem = '#'+semantic_flag+' \n'+cleaner.cleaned_context+'\n'
            out = half_cleaned_sem+'\n'
            return out
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CleanReforgeAmrSemantic]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CleanReforgeTreeSemantic(self, semantic, semantic_flag):
        """
        This function allows to convert a raw AMR semantic (graph-) string
        into a cleaned and minimalized anytree reprenstation of it.
            :param semantic: raw semantic input
            :param semantic_flag: marker/delim to add to cleaned semantic
        """
        try:
            half_cleaned_sem = '#'+semantic_flag+' '+semantic+'\n'
            out = half_cleaned_sem+'\n'
            return self.t_parser.Execute(out, semantic_flag, self.is_showing_feedback, self.is_saving)
            #//TODO remove or use!
            #self.g_parser = GParser(out, self.extension_dict)
            #return self.g_parser.GetNetworkxGraph()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CleanReforgeTreeSemantic]. Arguments:\n{1!r}"
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
            #//TODO here check a semantic is okay with Cleaner.isCleaned => if not discard 
            sentence = sentence_flag+' '+sentence
            sentence = self.RemoveEnclosingAngleBracket(sentence)
            if(self.as_amr):
                semantic = self.CleanReforgeAmrSemantic(semantic, semantic_flag)
                return [sentence, semantic]
            else:
                semantic = self.CleanReforgeTreeSemantic(semantic, semantic_flag)
                return [sentence, semantic]
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

            extracted = self.extractor.Extract(dataset, 
                                               self.context_max_length, 
                                               self.constants.SENTENCE_DELIM, 
                                               self.constants.FILE_DELIM)        

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
            print('AMR? ', self.as_amr)
            print('Look_up: \n', self.extension_dict)

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
            outpath = writer.out_path
            writer.Store()

            if (self.is_showing_feedback): print('outpath: ', outpath)
            
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