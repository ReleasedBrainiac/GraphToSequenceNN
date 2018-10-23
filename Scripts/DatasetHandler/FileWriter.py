from DatasetHandler.ContentSupport import isStr, isNotNone, isList
from DatasetHandler.ContentSupport import setOrDefault
from Configurable.ProjectConstants import Constants

class Writer:

    # Variables init
    writer_encoding = 'utf8'
    path = None
    context = None
    dataset_pairs = None
    output_extender = None

    # Class init 
    constants = Constants()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, input_path, in_output_extender='output',data_pairs=None, in_context=None):
        if isNotNone(input_path) and isStr(input_path):
            self.path = input_path

        if isNotNone(data_pairs) and isList(data_pairs):
            self.dataset_pairs = data_pairs

        if isNotNone(in_context) and isStr(in_context):
            self.context = in_context

        if isNotNone(in_output_extender) and isStr(in_output_extender):
            self.output_extender = in_output_extender
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def SavingCorpus(self, data_pair):
        """
        This function build a simple concatenation string containing a sentence and a semantic.
            :param sentence: cleaned sentence with sentences flag
            :param semantic: cleaned correspondign semantic for the sentence with semantic flag
        """
        if isNotNone(data_pair) and isStr(data_pair[0]) and isNotNone(data_pair[1]):
                return data_pair[0] + data_pair[1]
        else:
            print('WRONG INPUT FOR [SavingCorpus]')
            return None

    def GetOutputPath(self):
        """
        This function return a result output path depending on the given input path and a extender.
        """
        try:
            return setOrDefault(self.path+'.'+ self.output_extender , self.constants.TYP_ERROR, isStr(self.output_extender))
        except ValueError:
            print('WRONG INPUT FOR [GetOutputPath]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def StoreContext(self):
        """
        This function saves a (stringified) context passed by class init into a given file.
        """
        try:
            with open(self.GetOutputPath(), 'w', encoding=self.writer_encoding) as fileOut:
                if isNotNone(self.context) and isStr(self.context):
                    fileOut.write(self.context)
                    fileOut.flush()

                print('Destination => ', self.path)
        except ValueError:
            print('WRONG INPUT FOR [StoreContext]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def StoreAMR(self):
        """
        This function save the collected content to a given file.
        """
        try:
            with open(self.GetOutputPath(), 'w', encoding=self.writer_encoding) as fileOut:
                for i in range(len(self.dataset_pairs)):
                    result = self.SavingCorpus(self.dataset_pairs[i])
                    if isNotNone(result):
                        fileOut.write(result)
                        fileOut.flush()

                print('Destination => ', self.path)
        except ValueError:
            print('WRONG INPUT FOR [StoreAMR]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)