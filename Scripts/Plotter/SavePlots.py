import matplotlib.pyplot as plt

class PlotSaver():
    """
    This class allow to save plots to images.
    """

    def __init__(self, model_description:str, pyplot_figure):
        """
        The class constructor. 
            :param model_description:str: something to name the image unique and is also the file name
        """ 
        try:
            self._model_description:str = model_description
            self._pyplot_figure = pyplot_figure
        except Exception as ex:
            template = "An exception of type {0} occurred in [PlotSaver.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SavePyPlotToFile(self, extender:str = None, orientation:str = 'landscape', image_type:str = 'png'):
        """
        This function is a simple wrapper for the PyPlot savefig function with default values.
            :param extender:str: extender for the filename [Default None]
            :param orientation:str: print orientation [Default 'landscape']
            :parama image_type:str: image file type [Default 'png']
        """   
        try:
            if extender is None:
                self._pyplot_figure.savefig((self._model_description+'plot.'+image_type), orientation=orientation)
            else: 
                self._pyplot_figure.savefig((self._model_description+extender+'.'+image_type), orientation=orientation)
        except Exception as ex:
            template = "An exception of type {0} occurred in [PlotSaver.SavePyPlotToFile]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)