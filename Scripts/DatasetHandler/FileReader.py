import re
import json
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

    def Lines(self):
        """
        This function provides a file lines reader.
        """
        try:
            with open(self.path, 'r', encoding="utf8") as fileIn:
                return fileIn.readlines()
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.Lines]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LineReadContent(self):
        """
        This function provides a file reader for the AMR look up table.
        """
        try:
            look_up_elements = {}
            for line in self.Lines():
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

    def LoadJson(self):
        """
        This method parse a json file containing a list of amr and string pairs. 
        """
        try:
            sentence_lengths:list = []
            semantic_lengths:list = []
            pairs:list = []

            with open(self.path, 'r+') as f:
                jsons = json.load(f)
                for elem in jsons:
                    amr:str = elem['amr']
                    sent:str = elem['sent']

                    if isNotNone(amr) and isNotNone(sent):
                        semantic_lengths.append(len(amr))
                        sentence_lengths.append(len(amr))
                        pairs.append([sent, amr])

                return sentence_lengths, semantic_lengths, pairs
        except Exception as ex:
            template = "An exception of type {0} occurred in [FileReader.LoadJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)