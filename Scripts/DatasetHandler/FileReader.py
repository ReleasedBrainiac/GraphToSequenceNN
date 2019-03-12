from DatasetHandler.ContentSupport import isNotNone

class Reader():
    """
    This class provides a FileReader for text containing files with an [otpional] delimiter.
    """
    path = None
    delimiter = '#'

    def __init__(self, input_path:str =None, delimiter:str ='#'):
        """
        The class constructor check for valid input and store it for local usage. 
            :param input_path: path of file with string content
            :param delimiter: an optional sign that allow to split an amr dataset at each occurence
        """   
        try:
            if isNotNone(input_path): self.path = input_path
            if isNotNone(delimiter): self.delimiter = delimiter
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LineReadContent(self):
        """
        This function provides a file reader for the AMR look up table.
        """
        try:
            look_up_elements = {}
            with open(self.path, 'r', encoding="utf8") as fileIn:
                data=fileIn.readlines()
                for line in data:
                    content = line.replace('\n','').replace(',','').split('#')
                    look_up_elements[content[0]]=content[1]

            return look_up_elements
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.LineReadContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GroupReadAMR(self):
        """
        This function provides a file reader for the AMR dataset.
        """
        try:
            with open(self.path, 'r', encoding="utf8") as fileIn:
                return fileIn.read().split(self.delimiter)
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.GroupReadAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)