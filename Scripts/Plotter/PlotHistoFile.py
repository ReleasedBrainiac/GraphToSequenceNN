import sys

class HistoryPlotter(object):
    """
    This class provides a FileReader for text files.
    """

    ENCODING:str = sys.stdout.encoding or sys.getfilesystemencoding()

    def __init__(self, histo_context:str = None):
        """
        The class constructor. 
            :param histo_context:str: history context of the hitory file
        """   
        try:
            
            self._histo_context = histo_context  if not (histo_context is None) else None           
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)