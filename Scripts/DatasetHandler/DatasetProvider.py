import re
from DatasetHandler.ContentSupport import isStr, isInt, isNotNone, setOrDefault, CalculateMeanValue
from DatasetHandler.FileReader import Reader
from DatasetHandler.FileWriter import Writer
from DatasetHandler.DatasetExtractor import Extractor
from Configurable.ProjectConstants import Constants
from TreeHandler.TreeParser import TParser
from GraphHandler.SemanticMatrixBuilder import MatrixBuilder as MParser
from AMRHandler.AMRCleaner import Cleaner


class DatasetPipeline():
    """
    This class prepare the AMR for further usage. 
    It is possible to only clean and store the AMR dataset or it can be cleaned and passed to other processes.

    Used Resources:
        => https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
        => https://www.pythonsheets.com/notes/python-rexp.html
    """

    look_up_extension_replace_path = './Datasets/LookUpAMR/supported_amr_internal_nodes_lookup.txt'
    extension_dict =  Reader(input_path=look_up_extension_replace_path).LineReadContent()

    def __init__(self, 
                 in_path:str=None, 
                 output_path_extender:str='ouput', 
                 max_length:int=-1, 
                 show_feedback:bool =False, 
                 keep_edges:bool =False, 
                 min_cardinality:int =1, 
                 max_cardinality:int =100):
        """
        This class constructor collect informations about the input and output files.
        Further its possible to define a max_lengt for the used dataset. 
        It defines the string length for sentences and doubled it for the semantics string length.
        Ever missmatch will be dropped out!
        If it is negative the module will use all dataset elements.

        Attention:
            With min_cardinality and max_cardinality you can define the used dataset after processing. 
            This allows to handle hughe differnces in the dataset groups 
                ~> depends on knowledge about "well defined datasets" rules

            :param in_path:str: dataset input path
            :param output_path_extender:str: result output path
            :param max_length:int: context length restriction
            :param show_feedback:bool: show process content as console feedback
            :param keep_edges:bool: include edges in the amr cleaner strategy
            :param min_cardinality:int: define min range for the node matrix representation [>2 (at least 3 nodes/words) depends on the SPO sentence definition in english]
            :param max_cardinality:int: define max range for the node matrix representation 
        """   
        try:
            self.constants = Constants()
            self.in_path = setOrDefault(in_path, self.constants.TYP_ERROR, isStr(in_path))
            self.dataset_drop_outs = 0
            self.max_observed_nodes_cardinality = 0
            self.set_unique_graph_node_cardinalities = set()
            self.list_graph_node_cardinalities = []
            self.count_graph_node_cardinalities_occourences = dict()
            self.is_showing_feedback = show_feedback
            self.is_saving = False
            self.is_keeping_edges = keep_edges
            self.as_amr = False
            self.out_path_extender = output_path_extender
            self.restriction_sentence = setOrDefault(max_length, -1, isInt(max_length))
            self.restriction_semantic = -1 if (max_length < 0)  else (2 * self.restriction_sentence)
            self.min_cardinality = min_cardinality if (min_cardinality > 2) else 3
            self.max_cardinality = max_cardinality if (max_cardinality >= min_cardinality) else 100
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.__init__]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RemoveEnclosingAngleBracket(self, in_sentence:str):
        """
        This method cleans up the a given amr extracted sentence from text formatting markup.
            :param in_sentence:str: raw sentence AMR split element 
        """
        try:
            in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
            in_sentence = re.sub('</[^>]+>','', in_sentence)
            return  re.sub('<[^/>]+/>','', '#'+in_sentence)+'\n'
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.RemoveEnclosingAngleBracket]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeAmrSemanticString(self, semantic:str):
        """
        This function converts a raw AMR semantic into a cleaned and minimalized version of it.
            :param semantic:str: raw semantic input
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

    def ForgeAmrTree(self, semantic:str):
        """
        This function converts a AMR semantic into a cleaned and minimalized anytree reprenstation.
            :param semantic:str: semantic string
        """
        try:
            return TParser(semantic, self.is_showing_feedback, self.is_saving).Execute()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrTree]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeMatrices(self, semantic:str):
        """
        This function converts a given semantic into a cleaned metrix representation [Neigbouring and Nodes].
            :param semantic:str: semantic string
        """   
        try:
            return MParser(context=semantic, show_feedback=self.is_showing_feedback).Execute()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeMatrices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasetPair(self, data_pair:list):
        """
        This function collect a data pair from raw sentences and semantics.
        ATTENTION: [as_amr = true] does not support conversion with ConvertToTensorMatrices!
            :param data_pair:list: amr data pair
        """
        try:
            sentence = self.RemoveEnclosingAngleBracket(self.constants.SENTENCE_DELIM+' '+data_pair[0]).replace('\n','')
            semantic = self.ForgeAmrSemanticString(data_pair[1])

            if(not self.as_amr): semantic = self.ForgeMatrices(semantic)
                
            if isNotNone(semantic) and isNotNone(sentence): 
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectDatasetPair]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectAllDatasetPairs(self, data_pairs:list):
        """
        This function collect multiples pairs of semantic and sentence data as list of data pairs.
        For this case we pass arrays of raw sentences and semantics, 
        where index i in both arrays point to a sentence and the corresponding semantic.
            :param data_pairs:list: array of amr data pairs
        """
        try:
            dataset_pairs_sent_sem = []
            for pair in data_pairs: 
                data_pair = self.CollectDatasetPair(pair)
                if(not self.as_amr):
                    edges_dim = data_pair[1][0][0].shape[0]

                    if (self.min_cardinality <= edges_dim and edges_dim <= self.max_cardinality):
                        self.list_graph_node_cardinalities.append(edges_dim)
                        self.set_unique_graph_node_cardinalities.add(edges_dim)
                        self.max_observed_nodes_cardinality = max(self.max_observed_nodes_cardinality, edges_dim)
                        dataset_pairs_sent_sem.append(data_pair)
                else:
                    dataset_pairs_sent_sem.append(data_pair)
            return dataset_pairs_sent_sem
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectAllDatasetPairs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Pipeline(self):
        """
        This function collect the cleaned sentences and the cleaned semantics will hae the following structure: 

            [For Saving]
            => AMR string representation IF [as_amr = True] otherwise AnyTree as JSON

            [For Processing]
            => Numpy.ndarrays [[Neighbourings_fw, Neighbourings_bw], Nodes/Words/Features]

            Return: List[sentence, semantics]
        """
        try:
            dataset = Reader(self.in_path).GroupReadAMR()
            dataset=dataset[1:len(dataset)]
            sentence_lengths, semantic_lengths, pairs = Extractor(  in_content=dataset, 
                                                                    sentence_restriction=self.restriction_sentence, 
                                                                    semantics_restriction=self.restriction_semantic).Extract()
            mean_value_sentences = CalculateMeanValue(str_lengths=sentence_lengths)
            mean_value_semantics = CalculateMeanValue(str_lengths=semantic_lengths)     
            data_pairs = self.CollectAllDatasetPairs(pairs)

            if (not self.as_amr):
                for key in self.set_unique_graph_node_cardinalities:
                    self.count_graph_node_cardinalities_occourences[key] = self.list_graph_node_cardinalities.count(key)

            if self.is_showing_feedback:
                print('[Size Restriction]: Sentence =', self.restriction_sentence, '| Semantic = ', self.restriction_semantic)
                print('[Size Mean]: Sentences =', mean_value_sentences, '| Semantics = ', mean_value_semantics)
                print('[Count]: Sentences = ', len(sentence_lengths), '| Semantics = ', len(semantic_lengths))
                print('[Path]: ', self.in_path)
                print('[Dropouts]:', self.dataset_drop_outs)

            return data_pairs
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SaveData(self, as_amr:bool):
        """
        This function calls the Pipeline and store the desired results.
            :param as_amr:bool: IF True => raw AMR string ELSE anytree json string
        """
        try:
            
            self.as_amr = as_amr
            self.is_saving = True
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
            self.is_saving = False
            datapairs = self.Pipeline()
            print('Result structure:\n\t=> [Sentence, EdgeArrays [Forward Connections, Backward Connections], OrderedNodeDict(Content)]')
            return datapairs
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ProvideData]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ShowNodeCardinalityOccurences(self):
        """
        This function provides an overview about the type and amount of found cardinalities in th Dataset.
        """
        try:
            print('Graph Node Cardinality Occourences:')
            for key in self.count_graph_node_cardinalities_occourences.keys():
                print("\t=> [", key, "] =",self.count_graph_node_cardinalities_occourences[key], "times")
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ShowNodeCardinalityOccurences]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)