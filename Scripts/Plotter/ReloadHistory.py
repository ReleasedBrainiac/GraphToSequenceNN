import sys, re
from DatasetHandler.FileReader import Reader
from DatasetHandler.ContentSupport import isNotNone

class HistoryLoader():
    """
    This class provides a history loader.
    It contains the context collection of console logged history callbacks.
    #TODO later also from pickle.dumps
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
    TIME_STEP_CLEAN_REX:str = r'([0-9]+[ms]* [0-9]+[ms]*\/step[ -]+)'
    
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

    def __init__(self, path:str = None):
        """
        The class constructor. 
            :param path:str: path of the file containing the history
        """ 
        try:
            self._path = path if isNotNone(path) else None
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryLoader.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def HandleBatchLines(self, line:str):
        """
        This method handles logged batch lines.
            :param line:str: 
        """   
        try:
            if re.search(self.ETA_CLEAN_REX, line):
                line = re.sub(self.ETA_CLEAN_REX, '', line)
                self._losses.append(re.search(self._sub_element_rex.format(self._sub_elements[0]),line).group(2))
                self._top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[1]),line).group(2))
                self._categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[2]),line).group(2))
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryLoader.HandleBatchLines]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def HandleEpochResultLines(self, line:str):
        """
        This method handles logged epoch end result lines.
            :param line:str: 
        """   
        try:
            if re.search(self.TIME_STEP_CLEAN_REX, line):
                line = re.sub(self.TIME_STEP_CLEAN_REX, '', line)
                self._losses.append(re.search(self._sub_element_rex.format(self._sub_elements[0]),line).group(2))
                self._top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[1]),line).group(2))
                self._categorical_accuracies.append(re.search(self._sub_element_rex.format(self._sub_elements[2]),line).group(2))
                self._val_losses.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[0]),line).group(2))
                self._val_top_k_categorical_accuracies.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[1]),line).group(2))
                self._val_categorical_accuracies.append(re.search(self._sub_element_rex.format(self.VAL_EXTENDER + self._sub_elements[2]),line).group(2))
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryLoader.HandleEpochResultLines]. Arguments:\n{1!r}"
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

                    self.HandleBatchLines(line)
                    self.HandleEpochResultLines(line)

            equal_train_size:bool = (len(self._losses) == len(self._top_k_categorical_accuracies) == len(self._categorical_accuracies))
            correct_ratio:bool = ((self._train_samples * self._epochen) == len(self._losses))
            equal_val_size:bool = (len(self._val_losses) == len(self._val_top_k_categorical_accuracies) == len(self._val_categorical_accuracies) == self._epochen)
            self._collector_validation = equal_train_size and equal_val_size and correct_ratio
        except Exception as ex:
            template = "An exception of type {0} occurred in [HistoryLoader.CollectContextFromLog]. Arguments:\n{1!r}"
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
            template = "An exception of type {0} occurred in [HistoryLoader.CollectingRespone]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)