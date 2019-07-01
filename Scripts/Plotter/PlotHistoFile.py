import sys, re
from DatasetHandler.FileReader import Reader

class HistoryPlotter(object):
    """
    This class provides a History file plotter using mathplot.
    It contains the context collection of console logged history callbacks.
    And allows to plot the result of this or of a given history.
    """

    TRAIN_FLAG:str = 'train'
    VALIDATE_FLAG:str = 'validate'
    EPOCH_FLAG:str = 'epoch'
    ETA_FLAG:str = 'eta'
    VAL_EXTENDER:str = 'val_'
    TRAIN_REX:str = r'({}[a-zA-Z ]+)([0-9]+)([a-zA-Z ]+)'.format(TRAIN_FLAG)
    VALIDATE_REX:str = r'({}[a-zA-Z ]+)([0-9]+)([a-zA-Z ]+)'.format(VALIDATE_FLAG)
    EPOCHE_REX:str = r'^({}[ ]+)([0-9]+[\/])([0-9]+)$'.format(EPOCH_FLAG)
    ETA_CLEAN_REX:str = r'({}[ :]+)([0-9s:]+[ -]+)'.format(ETA_FLAG)
    EPS_STEP_CLEAN_REX:str = r'(^ *[0-9\/ ]+\[[.=>]*\][ -]+)'
    TIME_STEP_CLEAN_REX:str = r'([0-9]+s [0-9]+ms\/step[ -]+)'
    
    _sub_elements:list = ['loss', 'top_k_categorical_accuracy', 'categorical_accuracy']
    _sub_element_rex:str = r'(\b{}[: ]+)([0-9.]+)'

    _train_samples:int = -1
    _validate_samples:int = -1
    _epochen:int = -1
    _collector_validation:bool = False

    _losses:list = []
    _top_k_categorical_accuracies:list = []
    _categorical_accuracies:list = []
    _val_losses:list = []
    _val_top_k_categorical_accuracies:list = []
    _val_categorical_accuracies:list = []
    _eps_end_steps:list = []

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

    def CollectContextFromLog(self):
        """
        This method collect the necessary informations from console logged history callback.
        """   
        try:
            for line in Reader(self._path).Lines():
                line:str = line.lower()

                if (self.TRAIN_FLAG in line and self.VALIDATE_FLAG in line):
                    self._train_samples = int(re.search(self.TRAIN_REX, line).group(2))
                    self._validate_samples = int(re.search(self.VALIDATE_REX, line).group(2))
                
                eps_result = re.search(self.EPOCHE_REX, line)
                if eps_result:
                    self._epochen = int(eps_result.group(3))
            
                if re.search(self.EPS_STEP_CLEAN_REX, line):
                    line = re.sub(self.EPS_STEP_CLEAN_REX, '', line)
                    if re.search(self.ETA_CLEAN_REX, line):
                        line = re.sub(self.ETA_CLEAN_REX, '', line)
                        self._losses.append(re.search(self._sub_element_rex.format(self._sub_elements[0]),line).group(2))
                        self._top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[1]),line).group(2))
                        self._categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[2]),line).group(2))
                        
                    if re.search(self.TIME_STEP_CLEAN_REX, line):
                        line = re.sub(self.TIME_STEP_CLEAN_REX, '', line)
                        self._losses.append(re.search(self._sub_element_rex.format(self._sub_elements[0]),line).group(2))
                        self._top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[1]),line).group(2))
                        self._categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[2]),line).group(2))

                        self._val_losses.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[0]),line).group(2))
                        self._val_top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[1]),line).group(2))
                        self._val_categorical_accuracies.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[2]),line).group(2))

            equal_train_size:bool = (len(self._losses) == len(self._top_k_categorical_accuracies) == len(self._categorical_accuracies))
            correct_ratio:bool = ((self._train_samples * self._epochen) == len(self._losses))
            equal_val_size:bool = (len(self._val_losses) == len(self._val_top_k_categorical_accuracies) == len(self._val_categorical_accuracies) == self._epochen)
            self._collector_validation = equal_train_size and equal_val_size and correct_ratio

        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectContextFromLog]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectingRespone(self):
        """
        This method shows an overview about the amount of collected data and a collected data validation.
        """   
        try:
            print("Epochen:\t\t", self._epochen)
            print("Train:\t\t\t", self._train_samples)
            print("Valid:\t\t\t", self._validate_samples)
            print("[Train] Losses\t\t", len(self._losses))
            print("[Train] Top k:\t\t", len(self._top_k_categorical_accuracies))
            print("[Train] Categorical:\t", len(self._categorical_accuracies))
            print("[Valid] Losses:\t\t", len(self._val_losses))
            print("[Valid] Top k:\t\t", len(self._val_top_k_categorical_accuracies))
            print("[Valid] Categorical:\t", len(self._val_categorical_accuracies))
            print("Validated:\t\t", self._collector_validation)
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryPlotter.CollectingRespone]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

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

    def plot_history(history):
        
    
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