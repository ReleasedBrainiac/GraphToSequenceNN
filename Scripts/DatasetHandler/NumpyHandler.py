import numpy as np
import re
import sys
from DatasetHandler.ContentSupport import isNotNone, AssertNotNone, StatusReport, ConcatenateNdArrays, RepeatNTimsNdArray
from Configurable.ProjectConstants import Constants

class NumpyDatasetHandler():
    """
    This class provide save and load pipeline for 3D numpy arrays.
    Inspired by => https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file 
    """
    def __init__(self, path:str, show_feedback:bool = True):
        """
        This constructor collects the path string.
            :param path:str: path string for load or save
            :param show_feedback:bool: show reports and informations
        """
        AssertNotNone(path, "Given path for NumpyDatasetHandler constructor is none!")
        self._path:str = path
        self._show_feedback = show_feedback
        self._np_file_names:list = Constants().NP_TEACHER_FORCING_FILE_NAMES
        self._shape_regex:str = Constants().NP_GATHER_LOAD_SHAPE_REX
        
    def Save3DNdArrayToTxt(self, array:np.ndarray, report_steps:int = 1000, format:str = '%-7.2f'):
        """
        This method allow to save 3D numpy array with a optional format and extension into a given file in text form.
            :param array:np.ndarray: 3D numpy array
            :param report_steps:int: report after x times of seen slices
            :param format:str='%-7.2f': optional format. Default is writing the values in left-justified columns 7 characters in width an 2 decimal places.
        """
        try:

            AssertNotNone(array, "Save 3D numpy array input was none!")
            with open(self._path, 'w+') as outfile:
                outfile.write('# Array shape: {0}\n'.format(array.shape))

                dataset_len = len(array)
                if (self._show_feedback): print("Start saving array to file [",self._path, "]")

                for index in range(dataset_len):
                    outfile.write('# '+ str(index) +' slice\n')
                    np.savetxt(outfile, array[index], fmt=format)
                    if (self._show_feedback): StatusReport(run_index=index, max_index=dataset_len, steps=report_steps)

        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetHandler.Save3DNdArrayToTxt]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def Save2DNdArrayToTxt(self, array:np.ndarray, format:str = '%-7.2f'):
        """
        This method allow to save 2D numpy array with a optional format and extension into a given file in text form.
            :param array:np.ndarray: 2D numpy array
            :param format:str='%-7.2f': optional format. Default is writing the values in left-justified columns 7 characters in width an 2 decimal places.
        """
        try:
            AssertNotNone(array, "Save 2D numpy array input was none!")
            with open(self._path, 'w+') as outfile:
                outfile.write('# Array shape: {0}\n'.format(array.shape))
                np.savetxt(outfile, array, fmt=format)
        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetHandler.Save2DNdArrayToTxt]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def LoadNdArray(self, path:str = None):
        """
        This method auto load a numpy nd array txt file which was saved with SaveNdArrayToTxt!
            :param path:str: this is a optional path instead of the initial path from constructor.
        """
        try:
            path = path if isNotNone(path) else self._path

            f = open(path)
            shape_line = f.readline()
            f.close()

            shape_match = re.match(self._shape_regex, shape_line)
            shape_str = shape_match.group(2)

            if not ',)' in shape_str:
                shape_tupe = tuple(map(int, shape_str[1:-1].split(', ')))
                return np.loadtxt(path).reshape(shape_tupe)
            else:
                shape_tupe = tuple(map(int, shape_str[1:-2].split(', ')))
                return np.loadtxt(path).reshape(shape_tupe)
        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetHandler.Load3DNdArray]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    #TODO: This could be used for the fit generator as generator pipe -> needs more attention and changes!
    def LoadTeacherForcingDS(self):
        """
        This method allow to load the teacher forcing dataset parts.
        There specification is defined at the ProjectConstants.py
        Additional nice resource => https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions
        """
        try:
            dataset = []
            files_len = len(self._np_file_names)
            for index in range(files_len):
                cur_path = (self._path + self._np_file_names[index])
                print("Load File: ", cur_path)
                dataset.append(self.LoadNdArray(path = cur_path))
                if (self._show_feedback): 
                    StatusReport(run_index=index, max_index=files_len, steps=1)
                    print("Shape of ", self._np_file_names[index], " => ", dataset[index].shape)
            return dataset
        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetHandler.LoadTeacherForcingDS]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

class NumpyDatasetPreprocessor():
    """
    This class provide a teacher forcing dataset preprocessor.
    Additionally a network input preparation method called NetworkInputPreparation is provided.
    """
    def __init__(self, folder_path:str, show_feedback:bool = True):
        """
        This constructor collects the path string.
            :param _folder_path:str: folder path to save the datasets if its desired, names are autoset
            :param show_feedback:bool: show reports and informations
        """
        self._folder_path:str = folder_path
        self._show_feedback:bool = show_feedback
        self._np_file_names:list = Constants().NP_TEACHER_FORCING_FILE_NAMES

    def PreprocessTeacherForcingDS(self, nodes_embedding:np.ndarray, fw_look_up:np.ndarray, bw_look_up:np.ndarray, vecs_input_sentences:np.ndarray, vecs_target_sentences:np.ndarray, save:bool = False, use_padded_vecs:bool = True):
        """
        This method preprocess the dataset to provide astructure for teacher forcing -> sentences will be split into words -> generate sample for each word instead each sentence.
            :param nodes_embedding:np.ndarray: node embedding numpy array
            :param fw_look_up:np.ndarray: forward look up numpy array
            :param bw_look_up:np.ndarray: backward look up numpy array
            :param vecs_input_sentences:np.ndarray: input sentences vector numpy array -> they will be word level seperated
            :param vecs_target_sentences:np.ndarray: target sentences vector numpy array
            :param save:bool: save result or not
            :param use_padded_vecs:bool: the given sentence input and targets are only padded vecs not embedded
        """
        try:
            dataset_len = len(vecs_input_sentences)
            if len(nodes_embedding) == len(fw_look_up) == len(bw_look_up) == dataset_len == len(vecs_target_sentences):        

                nodes_emb = []
                forward_look_up = []
                backward_look_up = []
                vecs_input_words = []
                vecs_target_words = []

                print("Shape Inp: ", vecs_input_sentences.shape, " ... [0]", vecs_input_sentences[0])
                print("Shape Out: ", vecs_target_sentences.shape, " ... [0]",  vecs_target_sentences[0])

                print("Start Dataset Generator")
                for s_idx in range(dataset_len):

                    if use_padded_vecs:
                        tmp_vecs_input_words = np.trim_zeros(vecs_input_sentences[s_idx])
                        tmp_vecs_target_words = np.trim_zeros(vecs_target_sentences[s_idx])
                        qualified_entries = np.count_nonzero(tmp_vecs_input_words) - 1      #The -1 mean i will not include the the first copy since we keep the initial as well!
                    else:
                        tmp_vecs_input_words = vecs_input_sentences[s_idx]
                        tmp_vecs_target_words = vecs_target_sentences[s_idx]
                        qualified_entries = len(tmp_vecs_input_words) - 1

                    #print("tmp_vecs_input_words: ", tmp_vecs_input_words.shape)
                    #print("tmp_vecs_target_words: ", tmp_vecs_target_words.shape)
                    #print("qualified_entries: ", qualified_entries)

                    nodes_emb.append(RepeatNTimsNdArray(times=qualified_entries, array=nodes_embedding[s_idx]))
                    forward_look_up.append(RepeatNTimsNdArray(times=qualified_entries, array=fw_look_up[s_idx]))
                    backward_look_up.append(RepeatNTimsNdArray(times=qualified_entries, array=bw_look_up[s_idx]))

                    #print("Trimmed: ", tmp_vecs_input_words.reshape((tmp_vecs_input_words.shape[0],))[:-1])

                    vecs_input_words.append( tmp_vecs_input_words.reshape((tmp_vecs_input_words.shape[0],))[:-1] )


                    vecs_target_words.append( tmp_vecs_target_words.reshape((tmp_vecs_target_words.shape[0],)) )

                    if (self._show_feedback): StatusReport(run_index=s_idx, max_index=dataset_len, steps=500)

                nodes_emb = ConcatenateNdArrays(nodes_emb)
                forward_look_up = ConcatenateNdArrays(forward_look_up)
                backward_look_up = ConcatenateNdArrays(backward_look_up)
                vecs_input_words = ConcatenateNdArrays(vecs_input_words)
                vecs_target_words = ConcatenateNdArrays(vecs_target_words)

                if save:
                    NumpyDatasetHandler(path=(self._folder_path + self._np_file_names[0])).Save3DNdArrayToTxt(array=nodes_emb)
                    NumpyDatasetHandler(path=(self._folder_path + self._np_file_names[1])).Save3DNdArrayToTxt(array=forward_look_up)
                    NumpyDatasetHandler(path=(self._folder_path + self._np_file_names[2])).Save3DNdArrayToTxt(array=backward_look_up)

                    NumpyDatasetHandler(path=(self._folder_path + self._np_file_names[3])).Save2DNdArrayToTxt(array=vecs_input_words)
                    NumpyDatasetHandler(path=(self._folder_path + self._np_file_names[4])).Save2DNdArrayToTxt(array=vecs_target_words)

                return nodes_emb, forward_look_up, backward_look_up, vecs_input_words, vecs_target_words
            else:
                assert (len(nodes_embedding) == len(fw_look_up) == len(bw_look_up) == dataset_len == len(vecs_target_sentences)), "The given inputs of GenerateDatasetTeacherForcing aren't machting at first dimension!"
        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetPreprocessor.GenerateDatasetTeacherForcing]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def NetworkInputPreparation(self, nodes_embedding:np.ndarray, fw_look_up:np.ndarray, bw_look_up:np.ndarray, vecs_input_words:np.ndarray, vecs_target_words:np.ndarray, split_border:int):
        """
        This method return the train and test data.
            :param nodes_embedding:np.ndarray: node embedding numpy array
            :param fw_look_up:np.ndarray: forward look up numpy array
            :param bw_look_up:np.ndarray: backward look up numpy array
            :param vecs_input_sentences:np.ndarray: input sentences vector numpy array -> they will be word level seperated
            :param vecs_target_sentences:np.ndarray: target sentences vector numpy array
            :param split_border:int: the index where to split the dataset
        """
        try:
            assert (len(nodes_embedding) == len(fw_look_up) == len(bw_look_up) == len(vecs_input_words) == len(vecs_target_words)), "The given inputs of NetworkInputPreparation aren't machting at first dimension!"
            assert (len(nodes_embedding) >= split_border), ("The split index value was to high! [", len(nodes_embedding), " >= ", split_border, "]")

            train_x = [ nodes_embedding[:split_border], 
                        fw_look_up[:split_border], 
                        bw_look_up[:split_border],
                        vecs_input_words[:split_border]]

            test_x = [  nodes_embedding[split_border:], 
                        fw_look_up[split_border:], 
                        bw_look_up[split_border:],
                        vecs_input_words[split_border:]]

            train_y = vecs_target_words[:split_border]
            test_y = vecs_target_words[split_border:]

            return train_x, train_y, test_x, test_y
        except Exception as ex:
            template = "An exception of type {0} occurred in [NumpyDatasetPreprocessor.NetworkInputPreparation]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)