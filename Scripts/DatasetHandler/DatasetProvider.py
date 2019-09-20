import re
from multiprocessing import Pool
from DatasetHandler.ContentSupport import isStr, isInt, isNotNone, setOrDefault, CalculateMeanValue
from DatasetHandler.FileReader import Reader
from DatasetHandler.FileWriter import Writer
from DatasetHandler.DatasetExtractor import Extractor
from Configurable.ProjectConstants import Constants
from GraphHandler.SemanticMatrixBuilder import MatrixBuilder as MParser
from AMRHandler.AMRCleaner import Cleaner
from Plotter.PlotBarChart import BarChart


class DatasetPipeline:
    """
    This class prepare the AMR for further usage. 
    It is possible to only clean and store the AMR dataset or it can be cleaned and passed to other processes.

    Used Resources:
        => https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
        => https://www.pythonsheets.com/notes/python-rexp.html
    """
    
    def __init__(self, 
                 in_path:str=None, 
                 output_path_extender:str='ouput', 
                 max_length:int=-1, 
                 show_feedback:bool =False, 
                 keep_edges:bool =False, 
                 min_cardinality:int =1, 
                 max_cardinality:int =100,
                 cpu_cores:int = 1,
                 saving_cleaned_data:bool = False,
                 stringified_amr:bool = False):
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
            :param cpu_cores:int: define the number of existing/accessible cpu cores.
            :param saving_cleaned_data:bool: allow to save the cleaned dataset.
            :param stringified_amr:bool: convert semantic to matrices
        """   
        try:
            self._constants = Constants()
            self._look_up_ext_rep_path = './Datasets/LookUpAMR/supported_amr_internal_nodes_lookup.txt'
            self._extension_dict =  Reader(path=self._look_up_ext_rep_path, seperator_regex=self._constants.MAPPING_SPLIT_REGEX).LineReadContent()
            self._in_path = setOrDefault(in_path, self._constants.TYP_ERROR, isStr(in_path))
            self._dataset_drop_outs = 0
            self._max_chars_sentences = 0
            self._max_words_sentences = 0
            self._max_chars_semantics = 0
            self._max_observed_nodes_cardinality = 0
            self._unique_graph_node_cardinalities = set()
            self._graph_node_cardinalities_list = []
            self._count_graph_node_cards_occs = dict()
            self._is_showing_feedback = show_feedback
            self._is_saving = saving_cleaned_data
            self._is_keeping_edges = keep_edges
            self._out_path_extender = output_path_extender
            self._restriction_chars_sentence = setOrDefault(max_length, -1, isInt(max_length))
            self._restriction_chars_semantic = -1 if (max_length < 0)  else (2 * self._restriction_chars_sentence)
            self._min_cardinality = min_cardinality if (min_cardinality > 2) else 3
            self._max_cardinality = max_cardinality if (max_cardinality >= min_cardinality) else 100
            self._cpu_cores = cpu_cores if (cpu_cores > 1) else 1
            self._stringified_amr = stringified_amr
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
            return re.sub('<[^/>]+/>','', '#'+in_sentence)+'\n'
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
            node_parenthesis = ['(',')'] if ('(' in semantic and ')' in semantic) else None
            cleaner = Cleaner(input_context=semantic, input_extension_dict=self._extension_dict, keep_edges=self._is_keeping_edges, node_parenthesis=node_parenthesis)

            if cleaner.isCleaned:
                self._extension_dict = cleaner.extension_dict
                return '#'+self._constants.SEMANTIC_DELIM+' \n'+cleaner.cleaned_context+'\n'+'\n'
            else:
                self._dataset_drop_outs += 1
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeAmrSemanticString]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ForgeMatrices(self, semantic:str):
        """
        This function converts a given semantic into a cleaned metrix representation [Neigbouring and Nodes].
            :param semantic:str: semantic string
        """   
        try:
            return MParser(context=semantic, show_feedback=self._is_showing_feedback).Execute()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ForgeMatrices]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectDatasetPair(self, data_pair:list):
        """
        This function collect a data_pair from raw sentences and semantics.
        ATTENTION: [as_amr = true] does not support conversion with ConvertToTensorMatrices!
            :param data_pair:list: amr data_pair
        """

        try:          
            sentence:str = self.RemoveEnclosingAngleBracket(self._constants.SENTENCE_DELIM+' '+data_pair[0]).replace('\n','')
            semantic:str = self.EncloseWrongFormattedAMR(data_pair[1])
            semantic = self.ForgeAmrSemanticString(semantic)

            if(not self._stringified_amr): 
                semantic = self.ForgeMatrices(semantic)
                
            if isNotNone(semantic) and isNotNone(sentence): 
                sentence = self._constants.START_SIGN + ' ' + sentence + ' ' + self._constants.END_SIGN
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectDatasetPair]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


    def HandleSingleDataPair(self, pair):
        """
        This method process a single dataset. It'll be passed to the AMR cleaner pipe, the length restriction will be bypassed and the cardinalities will be collected.
            :param pair: 
        """
        try:
            data_pair = self.CollectDatasetPair(pair)
            if isNotNone(data_pair):
                if(not self._stringified_amr):
                    edges_dim = data_pair[1][0][0].shape[0]
                    if (self._min_cardinality <= edges_dim and edges_dim <= self._max_cardinality):
                        self.CollectCardinalities(edges_dim)
                        if (self._max_chars_sentences < len(data_pair[0])): self._max_chars_sentences = len(data_pair[0])
                        if (self._max_words_sentences < len(data_pair[0].split(" "))): self._max_words_sentences = len(data_pair[0].split(" "))
                        return data_pair
                    else:
                        return None
                else:
                    return data_pair
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.HandleSingleDataPair]. Arguments:\n{1!r}"
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
            self._max_chars_sentences = 0
            dataset_pairs_sent_sem = []

            if self._cpu_cores > 1:
                with Pool(self._cpu_cores) as p:
                    data_pair = p.map(self.HandleSingleDataPair, data_pairs)
                    if isNotNone(data_pair): dataset_pairs_sent_sem.append(data_pair)
            else:
                for pair in data_pairs: 
                    data_pair = self.HandleSingleDataPair(pair)
                    if isNotNone(data_pair): dataset_pairs_sent_sem.append(data_pair)
            return dataset_pairs_sent_sem
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectAllDatasetPairs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectCardinalities(self, edge_dim:int):
        """
        This function adds informations about a node cardinality and check if this is a new max.
            :param edge_dim:int: given cardinality
        """   
        try:
            self._graph_node_cardinalities_list.append(edge_dim)
            self._unique_graph_node_cardinalities.add(edge_dim)
            self._max_observed_nodes_cardinality = max(self._max_observed_nodes_cardinality, edge_dim)
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectCardinalities]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectCardinalityOccurences(self):
        """
        This function collects all node cardinality occurences.
        """   
        try:
            for key in self._unique_graph_node_cardinalities:
                self._count_graph_node_cards_occs[key] = self._graph_node_cardinalities_list.count(key)
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.CollectCardinalityOccurences]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    def Pipeline(self):
        """
        This function collect the cleaned sentences and the cleaned semantics
            Return: List[sentence, semantics]
        """
        try:
            dataset = None
            sentence_lengths:list = None
            semantic_lengths:list = None
            pairs:list = None
            reader = Reader(self._in_path)

            if ".txt" in self._in_path:
                print('Loading txt file...')
                dataset = reader.GroupReadAMR()
                dataset=dataset[1:len(dataset)]
                sentence_lengths, semantic_lengths, pairs = Extractor(  in_content=dataset, 
                                                                        sentence_restriction=self._restriction_chars_sentence, 
                                                                        semantics_restriction=self._restriction_chars_semantic).Extract()
            else:
                print('Loading json file...')
                sentence_lengths, semantic_lengths, pairs = reader.LoadJson()

            mean_sentences = CalculateMeanValue(str_lengths=sentence_lengths)
            mean_semantics = CalculateMeanValue(str_lengths=semantic_lengths)
            data_pairs = self.CollectAllDatasetPairs(pairs)

            if (not self._stringified_amr): self.CollectCardinalityOccurences()

            print('\n~~~~~~~~~~~~~ Cleaning AMR ~~~~~~~~~~~~')
            print('[Size Restriction]:\t Sentence =', self._restriction_chars_sentence, '| Semantic = ', self._restriction_chars_semantic)
            print('[Size Mean]:\t\t Sentences =', mean_sentences, '| Semantics = ', mean_semantics)
            print('[Size Max Chars]:\t\t Sentences =', self._max_chars_sentences)
            print('[Count]:\t\t Sentences =', len(sentence_lengths), '| Semantics = ', len(semantic_lengths))
            print('[Extensions]:\t\t', len(self._extension_dict))
            print('[Path]:\t\t\t', self._in_path)
            print('[Dropouts]:\t\t', self._dataset_drop_outs)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            return data_pairs
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    def ProvideData(self):
        """
        This function calls the Pipeline and return the cleaned dataset for ANN usage.
        Additional the cleaned dataset can be stored!
        """
        try:
            datapairs = self.Pipeline()

            if(self._is_saving): 
                Writer(self._in_path, self._out_path_extender, datapairs)
                print('Finished storing process!')

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
            for key in self._count_graph_node_cards_occs.keys():
                print("\t=> [", key, "] =",self._count_graph_node_cards_occs[key], "times")
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.ShowNodeCardinalityOccurences]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def PlotCardinalities(self, path:str = None):
        """
        This method plot the cardinalities.
            :param path:str: path combined with filename for file image storing
        """
        try:
            BarChart(   dataset = self._count_graph_node_cards_occs, 
                        min_value = self._min_cardinality,
                        max_value = self._max_cardinality, 
                        title = 'Cardinalities Occurences', 
                        short_title = 'Cardinality', 
                        x_label = 'Cardinalities', 
                        y_label = 'Occourences',
                        path = path)
            return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.PlotCardinalities]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseWrongFormattedAMR(self, amr:str):
        """
        This method return a correct enclosed amr string.
            :param amr:str: input amr
        """   
        try: 
            left_side = (not '(' is amr[0])
            right_side = (not ')' is amr[-1])

            if left_side or right_side:
                if amr.count('(') == amr.count(')'):
                    amr = '( ' + amr + ' )'
                else:
                    if left_side:
                        amr = '( ' + amr
                    else: 
                        amr = amr + ' )'

            return amr
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetProvider.EncloseWrongFormattedAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)