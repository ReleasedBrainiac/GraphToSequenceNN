from DatasetHandler.ContentSupport import isStr, isInt, isNotNone
from Configurable.ProjectConstants import Constants

class Extractor:

    context = None
    constants = None

    def __init__(self, in_content=None):
        try:
            self.context = in_content
            self.constants = Constants()
        except Exception as ex:
            template = "An exception of type {0} occurred in [DatasetExtractor.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RestrictionCorpus(self, max_len, sentence, semantic):
        """
        This funtion check a sentence and semantic pair satisfy the size restictions.
            :param max_len: max allows length of a sentence
            :param sentence: the cleaned sentence
            :param semantic: the cleaned correspondign semantic for the sentence
        """
        if isInt(max_len) and isStr(sentence) and isNotNone(semantic):

            sen_size = max_len
            sem_size = max_len*2

            if (max_len < 1) or ((len(sentence) < (sen_size+1)) and (len(semantic) < (sem_size+1))):
                return [sentence, semantic]

        else:
            print('WRONG INPUT FOR [RestrictionCorpus]')
            return None

    def ExtractSentence(self, in_content, index):
        """
        This function extract the sentence element from AMR corpus element!
            :param in_content: raw amr string fragment from split of full AMR dataset string
            :param index: position where the sentence was found
        """
        raw_start_index = in_content[index].find(self.constants.SENTENCE_DELIM)+6
        sentence = in_content[index]
        sent_len = len(sentence)
        return sentence[raw_start_index:sent_len-1]

    def ExtractSemantic(self, in_content, index):
        """
        This function extract the semantics element from AMR corpus element!
            :param in_content: raw amr string fragment from split of full AMR dataset string
            :param index: position where the semantic was found
        """
        raw_content = in_content[index].split('.txt')
        raw_con_index = len(raw_content)-1
        return raw_content[raw_con_index]

    def Extract(self,max_len):
        """
        This function collect the AMR-String-Representation and the corresponding sentence from AMR corpus.
            :param in_content: raw amr string fragment from split of full AMR dataset string
            :param max_len: maximal allowed length of a sentence and semantics
        """
        if isNotNone(self.context):
            in_content = self.context
            sentence = ''
            semantic = ''
            result_pair = None
            sentence_found = False
            semantic_found = False
            sentence_lengths = []
            semantic_lengths = []
            pairs = []

            for index, elem in enumerate(in_content):
                
                if (self.constants.SENTENCE_DELIM in elem):
                    sentence = self.ExtractSentence(in_content, index)
                    sentence_found = True

                if (self.constants.FILE_DELIM in elem and sentence_found):
                    semantic = self.ExtractSemantic(in_content, index)
                    semantic_found = True

                if sentence_found and semantic_found:
                    result_pair = self.RestrictionCorpus(max_len, sentence, semantic)
                    sentence_found = False
                    semantic_found = False

                    if isNotNone(result_pair):
                        sentence_lengths.append(len(result_pair[0]))
                        semantic_lengths.append(len(result_pair[1]))
                        if isNotNone(result_pair[1]) and isNotNone(result_pair[1]): pairs.append(result_pair)
                        result_pair = []

            if(len(sentence_lengths) == len(semantic_lengths) and isNotNone(pairs)):
                return [sentence_lengths, semantic_lengths, pairs]
            else:
                print('WRONG OUTPUT FOR [ExtractContent]... Size of outputs dont match!')
                return None
        else:
            print('WRONG INPUT FOR [ExtractContent]')
            return None