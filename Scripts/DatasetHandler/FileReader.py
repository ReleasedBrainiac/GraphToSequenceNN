from DatasetHandler.ContentSupport import isList, isStr, isInStr, isInt, isBool, isNone, isNotNone

class Reader:

    def GetAmrDatasetAsList(self, path):
        """
        This function provide a file reader for the AMR dataset.
            :param path: path string to dataset text file
        """
        if(isStr(path)):
            with open(path, 'r', encoding="utf8") as fileIn:
                data=fileIn.read()
                content=data.split('#')
                return content
        else:
            print('WRONG INPUT FOR [DatasetAsList]')
            return None