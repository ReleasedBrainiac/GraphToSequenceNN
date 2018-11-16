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
from DatasetHandler.DatasetExtractor import Extractor
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

    def ForgeAmrSemanticString(self, semantic):
        """
        This function allows to convert a raw AMR semantic into a cleaned and minimalized version of it.
            :param semantic: raw semantic input
        """
        try:
            cleaner = Cleaner(input_context=semantic, input_extension_dict=self.extension_dict, keep_edges=self.is_keeping_edges)
            if cleaner.isCleaned:
                self.extension_dict = cleaner.extension_dict
                return '#'+self.constants.SEMANTIC_DELIM+' \n'+cleaner.cleaned_context+'\n'+'\n'
            else:
                self.dataset_drop_outs += 1
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrSemanticString]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeAmrTree(self, semantic):
        """
        This function allows to convert a raw AMR semantic
        into a cleaned and minimalized anytree reprenstation of it.
            :param semantic: raw semantic input
        """
        try:
            out = '#'+self.constants.SEMANTIC_DELIM+' '+semantic+'\n'+'\n'
            return self.t_parser.Execute(out, self.constants.SEMANTIC_DELIM, self.is_showing_feedback, self.is_saving)
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrTree]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasetPair(self, data_pair):
        """
        This function collect a data pair from raw sentences and semantics.
        ATTENTION: [as_amr = true] does not support conversion with ConvertToTensorMatrices!
            :param data_pair: amr data pair
        """
        try:
            sentence = self.RemoveEnclosingAngleBracket(self.constants.SENTENCE_DELIM+' '+data_pair[0])
            if(self.as_amr):
                semantic = self.ForgeAmrSemanticString(data_pair[1])
            else: 
                semantic = self.ForgeAmrTree(data_pair[1])
                
            if isNotNone(semantic) and isNotNone(sentence): 
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectDatasetPair]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectAllDatasetPairs(self, data_pairs):
        """
        This function collect multiples pairs of semantic and sentence data as list of data pairs.
        For this case we pass arrays of raw sentences and semantics, 
        where index i in both arrays point to a sentence and the corresponding semantic.
            :param data_pairs: array of amr data pairs
        """
        try:
            dataset_pairs_sent_sem = []
            for i in data_pairs: dataset_pairs_sent_sem.append(self.CollectDatasetPair(data_pairs[i]))
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

            sentence_lengths, semantic_lengths, pairs = Extractor(dataset).Extract(max_len=self.context_max_length)        

            mean_value_sentences = self.eval_Helper.CalculateMeanValue(sentence_lengths)
            mean_value_semantics = self.eval_Helper.CalculateMeanValue(semantic_lengths)     

            data_pairs = self.CollectAllDatasetPairs(pairs)

            print('Max restriction: ', self.context_max_length)
            print('Path input: ', self.in_path)
            print('Count sentences: ', len(sentence_lengths))
            print('Count semantics: ', len(semantic_lengths))
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