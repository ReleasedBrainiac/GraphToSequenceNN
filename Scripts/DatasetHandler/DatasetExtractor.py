from DatasetHandler.ContentSupport import isStr, isInt, isNotNone
from Configurable.ProjectConstants import Constants

class Extractor:

    context = None
    constants = None
    size_restriction = -1

    def __init__(self, in_content=None, in_size_restriction=-1):
        """
        This class constructor store all passed values to global placeholders.
            :param in_content: input context by default None
            :param in_size_restriction: sentence and semantics restriction by default -1
        """   
        try:
            self.context = in_content
            self.size_restriction = in_size_restriction
            self.constants = Constants()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RestrictionCorpus(self, sentence, semantic):
        """
        This funtion check a sentence and semantic pair satisfy the size restictions.
            :param sentence: the cleaned sentence
            :param semantic: the cleaned correspondign semantic for the sentence
        """
        try:
            if (self.size_restriction < 1) or ((len(sentence) < (self.size_restriction+1)) and (len(semantic) < (self.size_restriction+1))):
                return [sentence, semantic]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.RestrictionCorpus]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractSentence(self, in_content, index):
        """
        This function extract the sentence element from AMR corpus element!
            :param in_content: raw amr string fragment from split of full AMR dataset string
            :param index: position where the sentence was found
        """
        try:
            raw_start_index = in_content[index].find(self.constants.SENTENCE_DELIM)+6
            sentence = in_content[index]
            sent_len = len(sentence)
            return sentence[raw_start_index:sent_len-1]
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.ExtractSentence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractSemantic(self, in_content, index):
        """
        This function extract the semantics element from AMR corpus element!
            :param in_content: raw amr string fragment from split of full AMR dataset string
            :param index: position where the semantic was found
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