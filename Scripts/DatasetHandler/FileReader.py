import re
from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isNotNone

class Reader:
    """
    This class provides a FileReader for text containing files with an [otpional] delimiter.
    """

    def __init__(self, path:str =None, seperator_regex:str =None):
        """
        The class constructor check for valid input and store it for local usage. 
            :param path:str: path of file with string content
            :param seperator_regex:str: an optional regex string that allow to split an amr dataset at each occurence
        """   
        try:
            self.constants = Constants()
            self.path = path  if isNotNone(path) else None           
            self.seperator_regex = seperator_regex if isNotNone(seperator_regex) else self.constants.ELEMENT_SPLIT_REGEX
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
                    replaced = line.replace('\n','').replace(',','')
                    content = re.split(self.seperator_regex, replaced)
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
                return re.split(self.seperator_regex, fileIn.read())
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.GroupReadAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)