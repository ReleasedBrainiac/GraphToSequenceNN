from DatasetHandler.ContentSupport import isStr, isNone, isNotNone, isList
from DatasetHandler.ContentSupport import setOrDefault
from Configurable.ProjectConstants import Constants

class Writer:

    # Variables init
    writer_encoding = 'utf8'
    path = None
    context = None
    dataset_pairs = None
    out_path = None

    # Class init 
    constants = Constants()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, input_path, in_output_extender='output',data_pairs=None, in_context=None):
        """
        This is the constructor of the Writer class. 
        Necessary is the input path only. 
        Extender for the output path provide a default value.
        To provide 2 types of input you can set amr datapairs or various context of type string.
        If both is present the data pairs wil be preferred.
            :param input_path: path of the input file
            :param in_output_extender='output': extender to create output file from input path
            :param data_pairs=None: amr data pairs list like List<Array{sentence, semantic}>
            :param in_context=None: optional if no data pairs present use context
        """   
        try:
            if isNotNone(input_path) and isStr(input_path):
                self.path = input_path
                self.out_path = setOrDefault(self.path+'.'+ in_output_extender , self.constants.TYP_ERROR, isStr(in_output_extender))

            if isNotNone(data_pairs) and isList(data_pairs):
                self.dataset_pairs = data_pairs
                self.StoreAMR()

            if isNotNone(in_context) and isStr(in_context) and isNone(data_pairs):
                self.context = in_context
                self.StoreContext()
            
            print('Destination: ', self.out_path)

        except Exception as ex:
            template = "An exception of type {0} occurred in [FileWriter.__init__]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def SavingCorpus(self, data_pair):
        """
        This function build a simple concatenation string containing a sentence and a semantic.
            :param sentence: cleaned sentence with sentences flag
            :param semantic: cleaned correspondign semantic for the sentence with semantic flag
        """
        try:
            if isNotNone(data_pair) and isStr(data_pair[0]) and isNotNone(data_pair[1]):
                    return data_pair[0] + data_pair[1]
            else:
                return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileWriter.SavingCorpus]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def StoreContext(self):
        """
        This function saves a (stringified) context passed by class init into a given file.
        """
        try:
            with open(self.out_path, 'w', encoding=self.writer_encoding) as fileOut:
                if isNotNone(self.context) and isStr(self.context):
                    fileOut.write(self.context)
                    fileOut.flush()
        except ValueError:
            print('WRONG INPUT FOR [FileWriter.StoreContext]')
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileWriter.StoreContext]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def StoreAMR(self):
        """
        This function save the collected content to a given file.
        """
        try:
            with open(self.out_path, 'w', encoding=self.writer_encoding) as fileOut:
                for i in range(len(self.dataset_pairs)):
                    result = self.SavingCorpus(self.dataset_pairs[i])
                    if isNotNone(result):
                        fileOut.write(result)
                        fileOut.flush()
        except ValueError:
            print('WRONG INPUT FOR [FileWriter.StoreAMR]')
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileWriter.StoreAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)