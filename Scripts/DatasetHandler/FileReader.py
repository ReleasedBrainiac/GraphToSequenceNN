from DatasetHandler.ContentSupport import isStr, isNotNone

class Reader:

    path = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, input_path):
        if isNotNone(input_path) and isStr(input_path):
            self.path = input_path

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
        except ValueError:
            print('WRONG INPUT FOR [DatasetAsList]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GroupReadAMR(self):
        """
        This function provide a file reader for the AMR dataset.
            :param path: path string to dataset text file
        """
        try:
            with open(self.path, 'r', encoding="utf8") as fileIn:
                data=fileIn.read()
                content=data.split('#')
                return content
        except ValueError:
            print('WRONG INPUT FOR [DatasetAsList]')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)