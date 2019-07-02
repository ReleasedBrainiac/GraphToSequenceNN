

class HistoryPlotter(object):
    """
    This class provides a History plotting pipeline using mathplot.
    """

    def __init__(self, path:str = None, history = None):
        """
        The class constructor. 
            :param path:str: path of the file containing the history
        """ 
        try:
            self._path = path if not (path is None) else None
            self._history = history if not (history is None) else None
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    




    #dict_keys(['val_loss', 'val_categorical_accuracy', 'loss', 'categorical_accuracy', 'lr'])



    def CollectFromHistory(self):
        try:
            self._losses = [s for s in self._history.history.keys() if self._sub_elements[0] in s and 'val' not in s]
            self._val_losses = [s for s in self._history.history.keys() if self._sub_elements[0] in s and 'val' in s]

            #acc_list = [s for s in self._history.history.keys() if self._sub_elements[1] in s and 'val' not in s]
            #val_acc_list = [s for s in self._history.history.keys() if self._sub_elements[1] in s and 'val' in s]

            #acc_list = [s for s in self._history.history.keys() if self._sub_elements[1] in s and 'val' not in s]
            #val_acc_list = [s for s in self._history.history.keys() if self._sub_elements[1] in s and 'val' in s]


        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectFromHistory]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def plot_history(self):
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
    
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        
        ## Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def PlotHistoryOld(self, history):
            print(history.history.keys())

            if 'top_k_categorical_accuracy' in self._accurracy:
                plt.plot(history.history['top_k_categorical_accuracy'], color='blue', label='train')
                plt.plot(history.history['val_top_k_categorical_accuracy'], color='orange', label='validation')
                plt.title('Model Top k Categorical Accuracy')
                plt.ylabel('Top k Categorical Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self.SAVE_PLOTS: 
                    self.SavePyPlotToFile(extender='top_k_categoriacal_epoch_plot')
                else: 
                   plt.show()

            if 'categorical_accuracy' in self._accurracy:
                plt.plot(history.history['categorical_accuracy'], color='blue', label='train')
                plt.plot(history.history['val_categorical_accuracy'], color='orange', label='validation')
                plt.title('Model Categorical Accuracy')
                plt.ylabel('Categorical Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self.SAVE_PLOTS: 
                    self.SavePyPlotToFile(extender='categoriacal_epoch_plot')
                else: 
                   plt.show()

            if 'lr' in history.history.keys():
                plt.plot(history.history['lr'], color='green', label='learning rate')
                plt.title('Model Learning Rate')
                plt.ylabel('Learning Rate')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                if self.SAVE_PLOTS: 
                    self.SavePyPlotToFile(extender='learning_rate_epoch_plot')
                else: 
                   plt.show()

            plt.plot(history.history['loss'], color='blue', label='train')
            plt.plot(history.history['val_loss'], color='orange', label='validation')
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            if self.SAVE_PLOTS: 
                self.SavePyPlotToFile(extender='loss_epoch_plot')
            else: 
                plt.show()