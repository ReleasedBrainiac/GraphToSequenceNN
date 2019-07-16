import re
import matplotlib.pyplot as plt
from DatasetHandler.ContentSupport import isNotNone, isNone
from Plotter.SavePlots import PlotSaver

class HistoryPlotter(object):
    """
    This class provides a History plotting pipeline using mathplot.
    """

    _using_history:bool = False # This for a later implemented part of the tool
    _path:str = None
    _history = None
    _history_keys:dict = None
    _history_keys_list:list = None
    _losses:list = None
    _val_losses:list = None
    _acc_stdcc_list:list = None
    _val_acc_stdcc_list:list = None
    _acc_topkcc_list:list = None
    _val_acc_topkcc_list:list = None
    _learning_rates:list = None
    _epochs:int = 0

    #TODO: File history plotting is not yet implemented

    def __init__(self, model_description:str, path:str = None, history = None, save_it:bool = True, new_style:bool = False):
        """
        The class constructor. 
        Attention: File history plotting is not yet implemented!
            :param model_description:str: something to name the image unique and is also the file name
            :param path:str: path of a file containing a history
            :param history: a history
            :param save_it:bool: save the plot instead of showing
            :param new_style:bool: desired matplot lib standard or new style
        """ 
        try:
            self._model_description = model_description if isNotNone(model_description) else 'undescribed_model'

            if isNotNone(path) and isNone(history):
                self._path:str = path 
                self._using_history = False

            if isNotNone(history):
                self._history = history 
                self._history_keys = history.history.keys()
                self._history_keys_list = list(self._history_keys)
                self._using_history = True

            
            self._new_style:bool = new_style
            self._save_it:bool = save_it
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def PlotHistory(self):
        """
        Thise method allow to plot a history from log or directly a keras history. 
        """   
        try:
            if self._using_history:
                if self._new_style:
                    self.CollectFromHistory()
                    self.DirectPlotHistory()
                else:
                    self.OldPlotHistory()
            #else:
            #    self.PlotHistoryFromLog()
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.PlotHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    def CollectAccFromHistory(self, name:str, train_val_list:list):
        """
         This method collect the accuracy data from the history into 2 plceholder lists.
            :param name:str: name of the used acc metric
            :param train_val_list:list: 2 placeholder lists with order 0 = train and 1 = validation
        """   
        try:
            name = re.sub('val_', '', name)
            if name in self._history_keys:
                train_val_list[0] = [s for s in self._history_keys if (name == s)]
                train_val_list[1] = [s for s in self._history_keys if ('val_'+name == s)]
                if isNotNone(train_val_list[0]) and isNotNone(train_val_list[1]):
                    self._history_keys_list.remove(name)
                    self._history_keys_list.remove('val_'+name)
                    print("Found accuracy metrics in history!")
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectAccFromHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectLossFromHistory(self):
        """
        This method collect the loss metric data from the history.
        """   
        try:
            loss_val:str = 'loss'
            if loss_val in self._history_keys:
                self._losses = [s for s in self._history_keys if (loss_val == s)]

                

                self._val_losses = [s for s in self._history_keys if ('val'+loss_val in s)]
                self._epochs = len(self._history.epoch)

                if len(self._losses) == 0 or len(self._val_losses) == 0:
                    print('Loss is missing in history')
                    return 

                if isNotNone(self._losses) and isNotNone(self._val_losses):
                    self._history_keys_list.remove(loss_val)
                    self._history_keys_list.remove('val_'+loss_val)
                    print("Found losses in history!")
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectLossFromHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        
    def CollectLearningRatesFromHistory(self):
        """
        This method collect the learning rate metric data from the history.
        """   
        try:
            lr_val:str = 'lr'
            if lr_val in self._history_keys:
                self._learning_rates = [s for s in self._history_keys if (lr_val == s)]
                if isNotNone(self._learning_rates):
                    self._history_keys_list.remove(lr_val)
                    print("Found learning rates in history!")
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectLearningRatesFromHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectFromHistory(self):
        if self._using_history:
            try:
                self.CollectLossFromHistory()
                self.CollectLearningRatesFromHistory()
                self.CollectAccFromHistory(name=self._history_keys_list[0], train_val_list = [self._acc_stdcc_list, self._val_acc_stdcc_list])
                self.CollectAccFromHistory(name=self._history_keys_list[0], train_val_list = [self._acc_topkcc_list, self._val_acc_topkcc_list])
            except Exception as ex:
                template = "An exception of type {0} occurred in [HistoryPlotter.CollectFromHistory]. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
        else:
            print('No history initialized!')

    def DirectPlotHistory(self):
        """
        This method helps to plot a keras history containing losses, accuracy and possibly least learning rates.
        """   
        try:
            ## Loss
            self.AccOrLossPlot( fig_num = 1, 
                                title = 'Model loss', 
                                metric = 'loss', 
                                axis_labels = ['train', 'validation'], 
                                history_labels = ['Loss', 'Epoch'], 
                                extender = 'loss_epoch_plot',
                                train_val_lists = [self._losses, self._val_losses])

            ## Top k Categorical Crossentropy
            if ('top_k_categorical_accuracy' in self._history_keys) and isNotNone(self._acc_topkcc_list) and isNotNone(self._val_acc_topkcc_list):
                self.AccOrLossPlot( fig_num = 2, 
                                    title = 'Model Top k Categorical Accuracy', 
                                    metric = 'top_k_categorical_accuracy', 
                                    axis_labels = ['train', 'validation'], 
                                    history_labels = ['Top k Categorical Accuracy', 'Epoch'], 
                                    extender = 'top_k_categoriacal_epoch_plot',
                                    train_val_lists = [self._acc_topkcc_list, self._val_acc_topkcc_list])

            ## Categorical Crossentropy
            if 'categorical_accuracy' in self._history_keys and isNotNone(self._acc_stdcc_list) and isNotNone(self._val_acc_stdcc_list):
                self.AccOrLossPlot( fig_num = 3, 
                                    title = 'Model Categorical Accuracy', 
                                    metric = 'categorical_accuracy', 
                                    axis_labels = ['train', 'validation'], 
                                    history_labels = ['Categorical Accuracy', 'Epoch'], 
                                    extender = 'categoriacal_epoch_plot',
                                    train_val_lists = [self._acc_stdcc_list, self._val_acc_stdcc_list])
                
            if 'lr' in self._history_keys and isNotNone(self._learning_rates):
                self.LearningPlot(  fig_num = 4,
                                    title = 'Model Learning Rate')

        except Exception as ex:
                template = "An exception of type {0} occurred in [HistoryPlotter.DirectPlotHistory]. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

    def OldPlotHistory(self):
        """
        This method plot the history in the old way.
        """   
        try:
            self.AccOrLossPlot( fig_num = 1, 
                                title = 'Model loss', 
                                metric = 'loss', 
                                axis_labels = ['train', 'validation'], 
                                history_labels = ['Loss', 'Epoch'], 
                                extender = 'loss_epoch_plot')

            if 'top_k_categorical_accuracy' in self._history_keys:
                self.AccOrLossPlot( fig_num = 2, 
                                    title = 'Model Top k Categorical Accuracy', 
                                    metric = 'top_k_categorical_accuracy', 
                                    axis_labels = ['train', 'validation'], 
                                    history_labels = ['Top k Categorical Accuracy', 'Epoch'], 
                                    extender = 'top_k_categoriacal_epoch_plot')

            if 'categorical_accuracy' in self._history_keys:
                self.AccOrLossPlot( fig_num = 3, 
                                    title = 'Model Categorical Accuracy', 
                                    metric = 'categorical_accuracy', 
                                    axis_labels = ['train', 'validation'], 
                                    history_labels = ['Categorical Accuracy', 'Epoch'], 
                                    extender = 'categoriacal_epoch_plot')

            if 'lr' in self._history_keys:
                self.LearningPlot(  fig_num = 4,
                                    title = 'Model Learning Rate')

        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.OldPlotHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
    
    def AccOrLossPlot(self, fig_num:int, title:str, metric:str, axis_labels:list = ['train', 'validation'], history_labels:list = ['Metric', 'Epoch'], extender:str = '_epoch_plot', train_val_lists:list = None):
        """
        This method wrapp the plot creation for a single metric of the keras train history.
            :param fig_num:int: figure number
            :param title:str: figure title
            :param metric:str: desired metric
            :param axis_labels:list: axis labels 
            :param history_labels:list: history labels
            :param extender:str: plot file name extender
            :param train_val_lists:list: a list containing the train and validation list of a defined metric
        """
        try:
            figure = plt.figure(fig_num)
            plt.suptitle(title, fontsize=14, fontweight='bold')

            if metric == 'loss': plt.title(self.CalcResultLoss(history=self._history))
            else: plt.title(self.CalcResultAccuracy(history=self._history, metric=metric))

            if not self._new_style:
                plt.plot(self._history.history[metric], color='blue', label=axis_labels[0])
                plt.plot(self._history.history['val_' + metric], color='orange', label=axis_labels[1])
            else:
                if (train_val_lists != None) and (len(train_val_lists) == 2):
                    for l in train_val_lists[0]: plt.plot(self._epochs, self._history.history[l], color='b', label='Training ' + metric + ' (' + str(format(self._history.history[l][-1],'.5f'))+')')
                    for l in train_val_lists[1]: plt.plot(self._epochs, self._history.history[l], color='g', label='Validation ' + metric + ' (' + str(format(self._history.history[l][-1],'.5f'))+')')

            plt.ylabel(history_labels[0])
            plt.xlabel(history_labels[1])
            plt.legend(axis_labels, loc='upper right')
            if self._save_it: 
                PlotSaver(self._model_description, figure).SavePyPlotToFile(extender=extender)
            else:
                plt.show()
            figure.clf()
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.AccOrLossPlot]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def LearningPlot(self, fig_num:int, title:str = 'Model Learning Rate', metric:str = 'lr', axis_labels:list = ['train', 'validation'], history_labels:list = ['Learning Rate', 'Epoch'], extender:str = 'learning_rate_epoch_plot'):
        """
        This method plot a the single learning rate curve.
            :param fig_num:int: figure number
            :param title:str: figure title
            :param metric:str: desired metric
            :param axis_labels:list: axis labels 
            :param history_labels:list: history labels
            :param extender:str: plot file name extender
        """
        try:
            figure = plt.figure(fig_num)
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.title(self.CalcResultLearnRate(history=self._history))

            if not self._new_style:
                plt.plot(self._history.history[metric], color='red', label='learning rate')
            else:
                for l in self._learning_rates: plt.plot(self._epochs, self._history.history[l], color='r', label='Learning Rate (' + str(format(self._history.history[l][-1],'.5f'))+')')
                

            plt.ylabel(history_labels[0])
            plt.xlabel(history_labels[1])
            plt.legend(axis_labels, loc='upper right')
            if self._save_it: 
                PlotSaver(self._model_description, figure).SavePyPlotToFile(extender='learning_rate_epoch_plot')
            else:
                plt.show()
            figure.clf()
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.LearningPlot]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CalcResultAccuracy(self, history, metric:str = 'acc'):
        """
        This method show the train acc results.
            :param history: history of the training
        """   
        try:
            return "Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*history.history[metric][-1], 100*history.history['val_'+metric][-1])
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CalcResultAccuracy]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CalcResultLoss(self, history):
        """
        This method show the train loss results.
            :param history: history of the training
        """   
        try:
            return 'Training loss: '+ str(history.history['loss'][-1])[:-6] +' / Validation loss: ' + str(history.history['val_loss'][-1])[:-6]
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CalcResultLoss]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CalcResultLearnRate(self, history):
        """
        This method show the train learn rate.
            :param history: history of the training
        """   
        try:
            return 'Training Learn Rate: '+ str(history.history['lr'][-1])
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CalcResultLearnRate]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

