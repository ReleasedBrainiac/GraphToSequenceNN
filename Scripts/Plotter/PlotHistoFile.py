import sys
from DatasetHandler.FileReader import Reader


class HistoryPlotter(object):
    """
    This class provides a History file plotter using mathplot.
    """

    _context_lines:list = None

    def __init__(self, path:str = None):
        """
        The class constructor. 
            :param path:str: path of the file containing the history
        """   
        try:
            self._path = path  if not (path is None) else None       
            if sys.path.exists(path):
                self._context_lines = Reader(path).Lines()
            else:
                print("File does not exist!")
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)