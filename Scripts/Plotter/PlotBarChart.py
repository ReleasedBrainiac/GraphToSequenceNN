import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from DatasetHandler.ContentSupport import isNotNone
from Plotter.SavePlots import PlotSaver

class BarChart(object):
    """
    This class allow to print simple bar charts.
    """
    def __init__(   self, 
                    dataset:dict, 
                    min_card:int, 
                    max_card:int, 
                    title:str = 'Cardinalities Occurences', 
                    short_title:str = 'Cardinality', 
                    y_label:str = 'Cardinalities', 
                    x_label:str = 'Occourences', 
                    path:str = None, 
                    save_it:bool = False):
        """
        The class constructor.
            :param dataset:dict: dataset as dict
            :param min_card:int: minimum cardinality
            :param max_card:int: maximum cardinality
            :param title:str: plot image long title
            :param short_title:str: short title
            :param y_label:str: label for the y values
            :param x_label:str: label for the x values
            :param path:str: path where to store the plot image
            :param save_it:bool: save or show plot
        """
        try:
            self._dataset:dict = dataset if isNotNone(dataset) else None
            self._min_card:int = min_card if (min_card > -1) else -1
            self._max_card:int = max_card if (max_card > -1) else -1
            self._title:str = title if isNotNone(title) else 'Untitled'
            self._short_title:str = short_title if isNotNone(short_title) else ''
            self._y_label:str = y_label if isNotNone(y_label) else 'Y_Value'
            self._x_label:str = x_label if isNotNone(x_label) else 'X_Value'
            self._path:str = path if isNotNone(path) else None
            self._save_it:bool = save_it

            if isNotNone(self._dataset):
                self.Print()
        except Exception as ex:
            template = "An exception of type {0} occurred in [BarChart.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Print(self):
        """
        This method print the bar chart.
        """
        try:
            print("aaaaa")
            i_card_subtitle:str = self._short_title + " Interval: [" + str(self._min_card) + " , " + str(self._max_card) + "]"
            ds_keys = self._dataset.keys()

            cardinalities:list = []
            occourences:list = []

            for key in ds_keys:
                cardinalities.append(key)
                occourences.append(self._dataset[key])

            print("baaaaaaaaaa")

            bc_fig = plt.figure()            
            plt.suptitle(self._title, fontsize=14, fontweight='bold')
            plt.title(i_card_subtitle)
            plt.bar(cardinalities, occourences)
            plt.ylabel(self._y_label)
            plt.xlabel(self._x_label)
            if self._save_it:
                PlotSaver(self._path, bc_fig).SavePyPlotToFile(extender=self._title.lower() + '_bar_chart')
            else:
                plt.show()

            bc_fig.clf()
        except Exception as ex:
            template = "An exception of type {0} occurred in [BarChart.Print]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)