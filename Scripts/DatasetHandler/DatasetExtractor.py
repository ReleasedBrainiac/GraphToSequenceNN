from DatasetHandler.ContentSupport import isNotNone
from Configurable.ProjectConstants import Constants

class Extractor:
    """
    This class allow to collect semantics and sentences from a given amr context.
    Additionally a size restrictions for the semantics an sentences can be passed.
    """

    def __init__(self, in_content:str =None, sentence_restriction:int =-1, semantics_restriction:int=-1):
        """
        This constructor store the given context, optional a size restriction and load at least the project constants.
            :param in_content:str: input context by default None
            :param sentence_restriction:int: sentence restriction by default -1
            :param semantics_restriction:int: semantic restriction by default -1
        """   
        try:
            self.context = in_content
            self.restriction_sentence = sentence_restriction
            self.restriction_semantic = semantics_restriction
            self.constants = Constants()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RestrictionCorpus(self, sentence:str, semantic:str):
        """
        This funtion check a sentence and semantic pair satisfy the size restiction.
            :param sentence:str: the cleaned sentence
            :param semantic:str: the cleaned correspondign semantic for the sentence
        """
        try:
            if (self.restriction_sentence < 1) or ((len(sentence) < (self.restriction_sentence+1)) and (len(semantic) < (self.restriction_semantic+1))):
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.RestrictionCorpus]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractSentence(self, in_content:str, index:int):
        """
        This function extract the sentence element from AMR corpus element.
            :param in_content:str: amr string fragment from AMR dataset string
            :param index:int: position where the sentence was found
        """
        try:
            raw_start_index = in_content[index].find(self.constants.SENTENCE_DELIM)+6
            sentence = in_content[index]
            return sentence[raw_start_index:len(sentence)-1]
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.ExtractSentence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractSemantic(self, in_content:str, index:int):
        """
        This function extract the semantics element from AMR corpus element.
            :param in_content:str: amr string fragment from AMR dataset string
            :param index:int: position where the semantic was found
        """
        try:
            raw_content = in_content[index].split('.txt')
            return raw_content[len(raw_content)-1]
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.ExtractSemantic]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Extract(self):
        """
        This function collect the AMR-String-Representation and the corresponding sentence from AMR corpus.
        """
        try:
            sentence = ''
            semantic = ''
            result_pair = None
            sentence_found = False
            semantic_found = False
            sentence_lengths = []
            semantic_lengths = []
            pairs = []

            for index, elem in enumerate(self.context):
                if (self.constants.SENTENCE_DELIM in elem):
                    sentence = self.ExtractSentence(self.context, index)
                    sentence_found = True

                if (self.constants.FILE_DELIM in elem and sentence_found):
                    semantic = self.ExtractSemantic(self.context, index)
                    semantic_found = True

                if sentence_found and semantic_found:
                    result_pair = self.RestrictionCorpus(sentence, semantic)
                    sentence_found = False
                    semantic_found = False

                    if isNotNone(result_pair):
                        sentence_lengths.append(len(result_pair[0]))
                        semantic_lengths.append(len(result_pair[1]))
                        if isNotNone(result_pair[1]) and isNotNone(result_pair[1]): pairs.append(result_pair)
                        result_pair = None

            if(len(sentence_lengths) == len(semantic_lengths) and isNotNone(pairs)):
                return [sentence_lengths, semantic_lengths, pairs]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.Extract]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)