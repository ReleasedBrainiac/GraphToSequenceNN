import re
import matplotlib.pyplot as plt
from DatasetHandler.ContentSupport import isNotNone
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
    _acc_list:list = None
    _val_acc_list:list = None
    _learning_rates:list = None

    def __init__(self, model_description:str, path:str = None, history = None,save_it:bool = True, new_style:bool = True):
        """
        The class constructor. 
            :param model_description:str: something to name the image unique and is also the file name
            :param path:str: path of the file containing the history
            :param history: a history
            :param new_style:bool: save the plot instead of showing
            :param new_style:bool: desired matplot lib standard or new style
        """ 
        try:
            self._model_description = model_description if isNotNone(model_description) else 'undescribed_model'

            if isNotNone(path):
                self._path = path 
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
                    self.DirectPlotHistory()
                else:
                    self.OldPlotHistory()
            else:
                self.PlotHistoryFromLog()
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectFromHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    def CollectAccFromHistory(self, name:str):
        """
         This method collect the accuracy data from the history.
            :param name:str: name of the used acc metric
        """   
        try:
            name = re.sub('val_', '', name)
            if name in self._history_keys:
                self._acc_list = [s for s in self._history_keys if (name == s)]
                self._val_acc_list = [s for s in self._history_keys if ('val_'+name == s)]
                if isNotNone(self._acc_list) and isNotNone(self._val_acc_list):
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
                self._val_losses = [s for s in self._history_keys ('val'+loss_val in s)]
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
                self.CollectAccFromHistory(name=self._history_keys_list[0])
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
            if len(self._losses) == 0:
                print('Loss is missing in history')
                return 

            epochs = range(1,len(self._history.history[self._losses[0]]) + 1)

            ## Loss
            loss_figure = plt.figure(1)
            for l in self._losses:
                plt.plot(epochs, self._history.history[l], color='b', label='Training loss (' + str(str(format(self._history.history[l][-1],'.5f'))+')'))
            for l in self._val_losses:
                plt.plot(epochs, self._history.history[l], color='g', label='Validation loss (' + str(str(format(self._history.history[l][-1],'.5f'))+')'))

            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            if self._save_it:
                    PlotSaver(self._model_description, loss_figure).SavePyPlotToFile(extender='loss_epoch_plot')

            ## Accuracy
            acc_figure = plt.figure(2)
            for l in self._acc_list:
                plt.plot(epochs, self._history.history[l], color='b', label='Training accuracy (' + str(format(self._history.history[l][-1],'.5f'))+')')
            for l in self._val_acc_list:    
                plt.plot(epochs, self._history.history[l], color='g', label='Validation accuracy (' + str(format(self._history.history[l][-1],'.5f'))+')')

            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            if self._save_it:
                    PlotSaver(self._model_description, acc_figure).SavePyPlotToFile(extender='accuracy_epoch_plots')

            if 'lr' in self._history_keys and isNotNone(self._learning_rates):
                lr_figure = plt.figure(3)
                for l in self._learning_rates:
                    plt.plot(epochs, self._history.history[l], color='r', label='Learning Rate (' + str(format(self._history.history[l][-1],'.5f'))+')')
                plt.title('Learning Rate')
                plt.xlabel('Epochs')
                plt.ylabel('Rate')
                plt.legend()

                if self._save_it:
                    PlotSaver(self._model_description, lr_figure).SavePyPlotToFile(extender='learning_rate_epoch_plot')
                plt.show()

        except Exception as ex:
                template = "An exception of type {0} occurred in [HistoryPlotter.DirectPlotHistory]. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

    def OldPlotHistory(self):
        """
        This method plot the history in the old way.
        """   
        try:
            loss_figure = plt.figure(1) 
            plt.suptitle('Model loss', fontsize=14, fontweight='bold')
            plt.title(self.CalcResultLoss(history=self._history))
            plt.plot(self._history.history['loss'], color='blue', label='train')
            plt.plot(self._history.history['val_loss'], color='orange', label='validation')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            if self._save_it: 
                PlotSaver(self._model_description, loss_figure).SavePyPlotToFile(extender='loss_epoch_plot')
            loss_figure.clf()

            if 'top_k_categorical_accuracy' in self._history_keys:
                acc_top_k_figure = plt.figure(2)
                plt.suptitle('Model Top k Categorical Accuracy', fontsize=14, fontweight='bold')
                plt.title(self.CalcResultAccuracy(history=self._history, metric='top_k_categorical_accuracy'))
                plt.plot(self._history.history['top_k_categorical_accuracy'], color='blue', label='train')
                plt.plot(self._history.history['val_top_k_categorical_accuracy'], color='orange', label='validation')
                plt.ylabel('Top k Categorical Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self._save_it:
                    PlotSaver(self._model_description, acc_top_k_figure).SavePyPlotToFile(extender='top_k_categoriacal_epoch_plot')
                acc_top_k_figure.clf()

            if 'categorical_accuracy' in self._history_keys:
                acc_figure = plt.figure(3)
                plt.suptitle('Model Categorical Accuracy', fontsize=14, fontweight='bold')
                plt.title(self.CalcResultAccuracy(history=self._history, metric='categorical_accuracy'))
                plt.plot(self._history.history['categorical_accuracy'], color='blue', label='train')
                plt.plot(self._history.history['val_categorical_accuracy'], color='orange', label='validation')
                plt.ylabel('Categorical Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self._save_it: 
                    PlotSaver(self._model_description, acc_figure).SavePyPlotToFile(extender='categoriacal_epoch_plot')
                acc_figure.clf()


            if 'lr' in self._history_keys:
                lr_figure = plt.figure(4)
                plt.suptitle('Model Learning Rate', fontsize=14, fontweight='bold')
                plt.title(self.CalcResultLearnRate(history=self._history))
                plt.plot(self._history.history['lr'], color='red', label='learning rate')
                plt.ylabel('Learning Rate')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self._save_it: 
                    PlotSaver(self._model_description, lr_figure).SavePyPlotToFile(extender='learning_rate_epoch_plot')
                lr_figure.clf()

            if not self._save_it: 
                plt.show()
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.OldPlotHistory]. Arguments:\n{1!r}"
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