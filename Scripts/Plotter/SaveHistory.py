import os, pickle
from DatasetHandler.ContentSupport import isNotNone

class HistorySaver():
    """
    This class allow to save a keras history to a file for later plotting.
    """

    _folder_path:str = None
    _file_path:str = None

    def __init__(self, folder_path:str, name:str, history):
        """
        This constructor collect inputs and create the file path.
            :param folder_path:str: folder where to store the history
            :param name: name of the history file
            :param history: the history
        """ 
        try:
            if isNotNone(folder_path):
                self._folder_path = folder_path
                self.MakeFolderIfMissing()
                path_seperator:str = '' if (folder_path[-1:] == '/') else '/'

                if isNotNone(name):
                    self._file_path = folder_path + path_seperator + name  
                    
                    if isNotNone(history):
                        self.SaveKerasHistory(history)

        except Exception as ex:
            template = "An exception of type {0} occurred in [HistorySaver.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def MakeFolderIfMissing(self):
        """
        This method creates a missing folder.
        """   
        try:
            if not os.path.exists(self._folder_path): 
                os.mkdir(self._folder_path)
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistorySaver.MakeFolderIfMissing]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SaveKerasHistory(self, history):
        """
        This method save a keras history to a file.
            :param history: the history
        """
        try:
            with open(self._file_path, 'wb') as history_file:
                pickle.dump(history.history, history_file)
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistorySaver.SaveKerasHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)