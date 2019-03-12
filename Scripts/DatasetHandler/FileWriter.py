from DatasetHandler.ContentSupport import isStr, isNotNone
from DatasetHandler.ContentSupport import setOrDefault, AssertNotNone
from Configurable.ProjectConstants import Constants

class Writer():
    """
    This class provides a FileWriter to store cleaned AMR Datasetpairs or stringyfied context.
    """
    writer_encoding = 'utf8'
    context = None
    dataset_pairs = None
    out_path = None
    

    def __init__(self, input_path:str, in_output_extender:str ='output',data_pairs:list =None, in_context:str =None):
        """
        This is the constructor of the (File-)Writer class. 
        Necessary is the input path only. 
        Extender for the output path provide a default value.
        To provide 2 types of input you can set amr datapairs or various context of type string.
        If both is present the data pairs wil be preferred.
            :param input_path:str: path of the input file
            :param in_output_extender:str: extender to create output file from input path
            :param data_pairs:list: amr data pairs list like List<Array{sentence, semantic}>
            :param in_context:str: optional if no data pairs present use context
        """   
        try:
            self.constants = Constants()
            self.path = input_path

            AssertNotNone(self.path, msg='Given path for FileWriter was None!')
            self.out_path = setOrDefault(self.path+'.'+ in_output_extender , self.constants.TYP_ERROR, isStr(in_output_extender))
            print('Destination: ', self.out_path)

            if isNotNone(data_pairs):
                self.dataset_pairs = data_pairs
                self.StoreAMR()
            else:
                AssertNotNone(in_context, msg='Given input for FileWriter was None!')
                self.context = in_context
                self.StoreContext()

        except Exception as ex:
            template = "An exception of type {0} occurred in [FileWriter.__init__]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SavingCorpus(self, data_pair:list):
        """
        This function build a simple concatenation string containing a sentence and a semantic.
            :param data_pair:list: a list containing pairs of cleaned sentence with sentences flags and cleaned corresponding semantics with semantics flags.
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
        This function save the collected amr content to a given file.
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