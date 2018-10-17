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
from GraphHandler.GraphTreeConverter import GTConverter
from AMRHandler.AMRCleaner import Cleaner

class DatasetPipelines:

    # Class inits
    extractor = Extractor()
    file_reader = Reader()
    file_writer = Writer()
    constants = Constants()
    eval_Helper = EvaluationHelpers()
    gt_converter = GTConverter()
    amr_cleaner = Cleaner()
    t_parser = TParser()

    def RemoveEnclosingAngleBracket(self, in_sentence):
        """
        This method clean up the a given amr extracted sentence from text formating markup.
            :param in_sentence: raw sentence AMR split element 
        """
        if(isStr(in_sentence)):
            in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
            in_sentence = re.sub('</[^>]+>','', in_sentence)
            in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
            return in_sentence+'\n'
        else:
            print('WRONG INPUT FOR [RemoveEnclosingAngleBracket]')
            return None

    def ReforgeSemanticToCleanMinimalARM(self, semantic, sem_flag):
        """
        This function allows to convert a raw AMR semantic (graph-) string  
        into a cleaned and minimalized version of it.
            :param semantic: raw semantic input
            :param sem_flag: marker/delim to add to cleaned semantic
        """
        if isStr(semantic) and isStr(sem_flag):
            store = semantic
            semantic = self.amr_cleaner.GenerateCleanAMR(semantic, '(', ')')

            if isNone(semantic):
                print(store)

            half_cleaned_sem = '#'+sem_flag+' \n'+semantic+'\n'
            out = half_cleaned_sem+'\n'
            return out
        else:
            print('WRONG INPUT FOR [ReforgeSemanticToCleanMinimalARM]')
            return None

    def ReforgeSemanticToMinimalAnyTree(self, semantic, sem_flag, print_console, to_process):
        """
        This function allows to convert a raw AMR semantic (graph-) string
        into a cleaned and minimalized anytree reprenstation of it.
            :param semantic: raw semantic input
            :param sem_flag: marker/delim to add to cleaned semantic
            :param print_console: If True you get console output for GraphSalvage.Execute
            :param to_process: If False you get a JsonString for saving in file.
        """
        if(isStr(semantic)) and (isStr(sem_flag)) and (isBool(print_console)) and (isBool(to_process)):
            half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
            out = half_cleaned_sem+'\n'
            return self.t_parser.Execute(out, sem_flag, print_console, to_process)
        else:
            print('WRONG INPUT FOR [ReforgeSemanticToMinimalAnyTree]')
            return None

    def GetSingleDatasetPair(self, sent_flag, sem_flag, sentence, semantic, want_as_arm, isShowConsole, isNotStoring):
        """
        This function collect a data pair from raw sentences and semantics.
        ATTENTION: [want_as_arm = true] does not support conversion with ConvertToTensorMatrices!
            :param sent_flag: a marker to attach to the cleaned sentence, make it easier to find later
            :param sem_flag: a marker to attach to the cleaned semantic, make it easier to find later
            :param sentence: raw input of the AMR sentence
            :param semantic: raw input of the AMR semantic
            :param want_as_arm: True then it return a tree-like formated AMR string for the semantic entry
            :param isShowConsole: True then it show in and out of the GraphSalvage.Cleaner on console
            :param isNotStoring: True you get GraphTree as AnyTree for further usage, else you get a JsonString for saving in file
        """
        if (isStr(sent_flag)) and (isStr(sem_flag)) and (isStr(sentence)) and (isStr(semantic)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
            sentence = sent_flag+' '+sentence
            sentence = self.RemoveEnclosingAngleBracket(sentence)
            if(want_as_arm):
                semantic = self.ReforgeSemanticToCleanMinimalARM(semantic, sem_flag)
                return [sentence, semantic]
            else:
                semantic = self.ReforgeSemanticToMinimalAnyTree(semantic, sem_flag, isShowConsole, isNotStoring)
                return [sentence, semantic]
        else:
            print('WRONG INPUT FOR [GetSingleDatasetPair]')
            return [None, None]

    def GetMultiDatasetPairs(self, sent_flag, sem_flag, sent_array, sem_array, want_as_arm, isShowConsole, isNotStoring):
        """
        This function collect multiples pairs of semantic and sentence data as list of data pairs.
        For this case we pass arrays of raw sentences and semantics, 
        where index i in both arrays point to a sentence and the corresponding semantic.
            :param sent_flag: a marker to attach to the cleaned sentence, make it easier to find later
            :param sem_flag: a marker to attach to the cleaned semantic, make it easier to find later
            :param sent_array: Array of the raw input of the AMR sentences
            :param sem_array: Array of the raw input of the AMR semantics
            :param want_as_arm: True then it return a tree-like formated AMR string for the semantic entry
            :param isShowConsole: True then it show in and out of the GraphSalvage.Cleaner on console
            :param isNotStoring: True you get GraphTree as AnyTree for further usage, else you get a JsonString for saving in file
        """
        if (isStr(sent_flag)) and (isStr(sem_flag)) and (isList(sent_array)) and (isList(sem_array)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
            dataset_pairs_sent_sem = []
            for i in range(min(len(sent_array), len(sem_array))):
                dataset_pairs_sent_sem.append(self.GetSingleDatasetPair(sent_flag, sem_flag, sent_array[i], sem_array[i], want_as_arm, isShowConsole, isNotStoring))

            return dataset_pairs_sent_sem
        else:
            print('WRONG INPUT FOR [GetMultiDatasetPair]')
            return []

    def BasicPipeline(self, inpath, output_extender, max_length, save_as_arm, print_console, is_not_saving):
        """
        This function collect the cleaned sentence graphs as:
        1. AMR string representation [save_as_arm = True]
        2. AnyTree [else]:
            1. as JSON     [is_not_saving = true]
            2. as AnyNode  [else]

            :param inpath: path of dataset text file
            :param output_extender: extender to define result filename
            :param max_length: max allows length for sentences
            :param save_as_arm: output will be save as tree like formated AMR string
            :param print_console: show all reforging at the Cleaner on console
            :param is_not_saving: set result to JSON or AnyNode [save_as_arm = False]
        """
        semantics  = []
        sentences  = []
        sents_lens = []
        sema_lens  = []

        max_length = setOrDefault(max_length, -1, isInt(max_length))
        inpath  = setOrDefault(inpath, self.constants.TYP_ERROR, isStr(inpath))

        dataset = self.file_reader.GetAmrDatasetAsList(inpath)
        dataset=dataset[1:len(dataset)]
        
        sents_lens, sema_lens, sentences, semantics = self.extractor.Extract(dataset, 
                                                                             max_length, 
                                                                             self.constants.SENTENCE_DELIM, 
                                                                             self.constants.FILE_DELIM)

        mw_value_sen = self.eval_Helper.CalculateMeanValue(sents_lens)
        mw_value_sem = self.eval_Helper.CalculateMeanValue(sema_lens)     

        data_pairs = self.GetMultiDatasetPairs(self.constants.SENTENCE_DELIM, 
                                               self.constants.SEMANTIC_DELIM, 
                                               sentences, 
                                               semantics, 
                                               save_as_arm, 
                                               print_console, 
                                               is_not_saving)

        print('Max restriction: ', max_length)
        print('Path input: ', inpath)
        print('Count sentences: ', len(sentences))
        print('Count semantics: ', len(semantics))
        print('Mean sentences: ', mw_value_sen)
        print('Mean semantics: ', mw_value_sem)

        return [mw_value_sen, mw_value_sem, max_length, data_pairs]

    def SavePipeline(self, inpath, output_extender, max_length, save_as_arm, print_console):
        """
        This function calls the BasicPipeline and store the desired results.
            :param inpath: path of dataset text file
            :param output_extender: extender to define result filename
            :param max_length: max allows length for sentences
            :param save_as_arm: output will be save as tree like formated AMR string
            :param print_console: show all reforging at the Cleaner on console
        """
        mw_value_sen, mw_value_sem, max_length, data_pairs = self.BasicPipeline(inpath, 
                                                                                output_extender, 
                                                                                max_length, 
                                                                                save_as_arm, 
                                                                                print_console, 
                                                                                False)

        outpath = self.file_writer.GetOutputPath(inpath, output_extender)

        if (print_console):
            print('outpath: ', outpath)

        self.file_writer.SaveToFile(outpath,
                                    mw_value_sen,
                                    mw_value_sem,
                                    max_length,
                                    data_pairs)
        
        return None

    def DataPipeline(self, inpath, output_extender, max_length, save_as_arm, print_console):
        """
        This function calls the BasicPipeline and return the cleaned dataset for ANN usage.
            :param inpath: path of dataset text file
            :param output_extender: extender to define result filename
            :param max_length: max allows length for sentences
            :param save_as_arm: output will be save as tree like formated AMR string
            :param print_console: show all reforging at the Cleaner on console
        """
        if(save_as_arm == True):
            print('Processing dataset on AMR string representation not supported! Please set [save_as_arm=FALSE]!')
            return None
        else:
            data_pairs = self.BasicPipeline(inpath, 
                                            output_extender, 
                                            max_length, 
                                            save_as_arm, 
                                            print_console, 
                                            True)[3]

            return self.gt_converter.GetDataSet(data_pairs)