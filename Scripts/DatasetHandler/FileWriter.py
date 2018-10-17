from DatasetHandler.ContentSupport import isStr, isNotNone
from DatasetHandler.ContentSupport import setOrDefault
from Configurable.ProjectConstants import Constants

class Writer:

    constants = Constants()

    def SavingCorpus(self, sentence, semantic):
        """
        This function build a simple concatenation string containing a sentence and a semantic.
            :param sentence: cleaned sentence with sentences flag
            :param semantic: cleaned correspondign semantic for the sentence with semantic flag
        """
        if isStr(sentence) and isNotNone(semantic):
                return sentence + semantic
        else:
            print('WRONG INPUT FOR [SavingCorpus]')
            return None

    def SaveToFile(self, path, len_sen_mw, len_sem_mw, max_len, data_pairs):
        """
        This function save the collected content to a given file.
            :param path: path to output file 
            :param len_sen_mw: mean of sentences length
            :param len_sem_mw: mean of semantics length
            :param max_len: max length of sentences we desire to store
            :param data_pairs: result data pairs as list
        """
        with open(path, 'w', encoding="utf8") as fileOut:
            for i in range(len(data_pairs)):
                result = self.SavingCorpus(data_pairs[i][0], data_pairs[i][1])
                if isNotNone(result):
                    fileOut.write(result)
                    fileOut.flush()

            print(path)
            return None

    def GetOutputPath(self, inpath, output_extender):
        """
        This function return a result output path depending on the given input path and a extender.
            :param inpath: raw data input path
            :param output_extender: result data path extender
        """
        if isStr(inpath) and isStr(output_extender):
            return setOrDefault(inpath+'.'+ output_extender , self.constants.TYP_ERROR, isStr(output_extender))
        else:
            print('WRONG INPUT FOR [GetOutputPath]')
            return None