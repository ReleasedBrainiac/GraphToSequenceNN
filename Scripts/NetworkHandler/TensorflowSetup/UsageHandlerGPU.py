import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from tensorflow.python.client import device_lib

class KTFGPUHandler():
    
    #TODO: missing class description

    ENV_THREAD_VAR:str = 'OMP_NUM_THREADS'
    DEVICE_TYPE:str = 'GPU'

    def __init__(self, gpu_fraction:float = 0.3):
        """
        This is the class constructor collecting the gpu fraction value.
            :param gpu_fraction:float: usage fraction
        """   
        try:
            self._num_threads = os.environ.get(self.ENV_THREAD_VAR)
            self._gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        except Exception as ex:
            template = "An exception of type {0} occurred in [KTFGPUHandler.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetSession(self):
        """
        This method allow to change the used GPUs number a percentage of a fixed fraction and return the changed session.
        Assume that you have 6GB of GPU memory and want to allocate ~2GB.
        """   
        try:
            if self._num_threads:
                return tf.Session(config=tf.ConfigProto(gpu_options=self._gpu_options, intra_op_parallelism_threads=self._num_threads))
            else:
                return tf.Session(config=tf.ConfigProto(gpu_options=self._gpu_options))
        except Exception as ex:
            template = "An exception of type {0} occurred in [KTFGPUHandler.GetSession]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def SetSession(self):
        """
        This method set the session with new GPU usage.
        """   
        try:
            KTF.set_session(self.GetSession())
        except Exception as ex:
            template = "An exception of type {0} occurred in [KTFGPUHandler.SetSession]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 

    def GetAvailableGPUs(self):
        """
        This method returns a list of available GPUs.
        """   
        try:
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == self.DEVICE_TYPE]
        except Exception as ex:
            template = "An exception of type {0} occurred in [KTFGPUHandler.GetAvailableGPUs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 
    
    def GetAvailableGPUsTF2(self, as_readable:bool = True):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')

            if as_readable:
                desc_gpus:list = []
                for gpu in gpus: desc_gpus.append(("[", gpu.name, " | ", gpu.device_type, "]"))
                gpus = desc_gpus

            return gpus

        except Exception as ex:
            template = "An exception of type {0} occurred in [KTFGPUHandler.GetAvailableGPUs]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 