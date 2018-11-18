from DatasetHandler.ContentSupport import isStr, isNotNone

class Reader:

    path = None
    delimiter = None

    def __init__(self, input_path, delimiter='#'):
        """
        The class constructor check for valid input. 
            :param input_path: path of file with string content
            :param delimiter: an optional sign that allow to split an amr dataset at each occurence
        """   
        try:
            if isNotNone(input_path) and isStr(input_path): self.path = input_path
            if isNotNone(delimiter) and isStr(delimiter): self.delimiter = delimiter
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LineReadContent(self):
        """
        This function provide a file reader for the AMR look up table.
            :param path: path string to dataset text file
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
            template = "An exception of type {0} occurred in [FileReader.DatasetAsList]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GroupReadAMR(self):
        """
        This function provide a file reader for the AMR dataset.
            :param path: path string to dataset text file
        """
        try:
            with open(self.path, 'r', encoding="utf8") as fileIn:
                return fileIn.read().split(self.delimiter)
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.GroupReadAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)