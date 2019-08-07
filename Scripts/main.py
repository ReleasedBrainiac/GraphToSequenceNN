import argparse
import os
import sys
import platform as pf
import numpy as np
from numpy import array_equal, argmax
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import activations
from keras.callbacks import History, ReduceLROnPlateau, BaseLogger

from time import gmtime, strftime
from Logger.Logger import FACLogger, FolderCreator
from Configurable.ProjectConstants  import Constants
from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import RoundUpRestricted, isNotNone, isNumber, CreateNListWithRepeatingValue, ConcatenateNdArray, RepeatNTimsNdArray
from GloVeHandler.GloVeDatasetPreprocessor import GloVeDatasetPreprocessor
from GloVeHandler.GloVeEmbedding import GloVeEmbedding
from DatasetHandler.FileWriter import Writer
from DatasetHandler.ContentSupport import MatrixExpansionWithZeros
from NetworkHandler.Builder.ModelBuilder import ModelBuilder
from Plotter.SaveHistory import HistorySaver
from Plotter.PlotHistory import HistoryPlotter
from NetworkHandler.TensorflowSetup.UsageHandlerGPU import KTFGPUHandler

#TODO bset tutorial => https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
#TODO many-to-many => https://github.com/keras-team/keras/issues/1029
#TODO another resource => https://data-science-blog.com/blog/2017/12/20/maschinelles-lernen-klassifikation-vs-regression/
#TODO IN MA => Code Next Level => https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py?source=post_page---------------------------
#TODO IN MA => Next Level 1 => https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
#TODO IN MA => Next Level 2 => http://proceedings.mlr.press/v48/niepert16.pdf
#TODO IN MA => Ausblick => https://github.com/philipperemy/keras-attention-mechanism
#TODO IN MA => Ausblick => https://github.com/keras-team/keras/issues/4962
#TODO IN MA => Code => Expansion of edge matrices why? => Layers weights!
#TODO IN MA => Code => Why min and max cardinality
#TODO IN MA => Code => https://keras.io/callbacks/ 
#TODO IN MA => Code => https://github.com/GeekLiB/keras/blob/master/tests/keras/test_callbacks.py
#TODO IN MA => Code => https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/
#TODO IN MA => Resource for MA and Code ~> https://stackoverflow.com/questions/32771786/predictions-using-a-keras-recurrent-neural-network-accuracy-is-always-1-0/32788454#32788454 
#TODO IN MA => Best LSTM resources ~> https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
#                                  ~> http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#TODO IN MA => 2nd best LSTM resource ~> https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#TODO IN MA => Kaggle Plot resource => https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search 

#TODO Guide to finalize Tool => https://keras.io/examples/lstm_stateful/
#TODO Guide to finalize Tool => https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/
#TODO Guide to finalize Tool => https://keras.io/models/model/#fit 
#TODO Guide to finalize Tool => https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
#TODO Guide to finalize Tool => https://keras.io/visualization/#training-history-visualization
#TODO Guide to finalize Tool => https://stackoverflow.com/questions/49910778/keras-sequence-to-sequence-model-loss-increases-without-bound

#TODO nice to use => https://stackoverflow.com/questions/45193744/keras-tensorflow-strange-results
#TODO nice to use => https://www.tensorflow.org/guide/summaries_and_tensorboard

class Graph2SeqInKeras():
    """
    This class provides a AMR Dataset cleaner which is especially developed for this tool 
    and it also provides a graph 2 sequence network construction with the Keras framework.

    The technological idea behind my code is partially based on the work => https://arxiv.org/abs/1804.00823
    The models hyperparamter definitions is oriented on the IBM tensorflow implementation => https://github.com/IBM/Graph2Seq

    #TODO If insertion of Philippe Remy's attention layer is necessary... change the next documentation lines.
    #TODO His repo => https://github.com/philipperemy/keras-attention-mechanism 

    Attention!
        This is NOT a reimplementation of the referenced repository.
        I strictly implemtented the idea of the paper for keras (currently without attention) users. 
        The interenal structure of the model is completely different to the ibm tensorflow implementation.
        In the first place i don't used an attention layer, since this tool only evaluates the possibility, 
        to implemtent the graph2sequence network structure in Keras and a model reach acc >= 50%.

        If there are any questions please feel free to open an issue on github.        
    """

    TF_CPP_MIN_LOG_LEVEL:str = '2'
    EPOCHS:int = 3
    VERBOSE:int = 1
    BATCH_SIZE:int = 1
    BUILDTYPE:int = 1
    DATASET_NAME:str = 'AMR Bio/amr-release-training-bio.txt' #'Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt' #
    fname = DATASET_NAME.split('/')[0]
    DATASET:str = './Datasets/Raw/'+DATASET_NAME
    GLOVE:str = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    GLOVE_VEC_SIZE:int = 100
    PLOT:str = "plot.png"
    EXTENDER:str = "dc.ouput"
    MAX_LENGTH_DATA:int = -1
    SHOW_FEEDBACK:bool = False
    SAVE_PLOTS = True
    SAVING_CLEANED_AMR:bool = False
    KEEP_EDGES:bool = True
    GLOVE_OUTPUT_DIM:int = 100
    GLOVE_VOCAB_SIZE:int = 5000
    VALIDATION_SPLIT:float = 0.2
    MIN_NODE_CARDINALITY:int = 15
    MAX_NODE_CARDINALITY:int = 35
    HOP_STEPS:int = 3
    SHUFFLE_DATASET:bool = True

    _accurracy:list = ['categorical_accuracy']
    _available_gpus = None
    _predict_percentage_split:float = 5.0
    _predict_split_value:int = -1
    _dataset_size:int = -1
    _history_keys:list = None

    # Run Switch
    MULTI_RUN = False

    # Single Run
    TIME_NOW:str = strftime("%Y%m%d %H_%M_%S", gmtime())
    FOLDERNAME:str = "graph2seq_model_" + fname + "_DT_" + TIME_NOW + "/"
    MODEL_DESC:str = FOLDERNAME + "model_" + fname + "_eps_"+ str(EPOCHS) + "_HOPS_" + str(HOP_STEPS) + "_GVSize_" + str(GLOVE_VEC_SIZE) + "_DT_" + TIME_NOW + "_"

    # Multi Run Setup!
    datasets = ['Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt', 'AMR Bio/amr-release-training-bio.txt']
    multi_epochs = [10, 15, 20, 10, 15, 20]
    multi_hops = [4, 5, 7, 4, 5, 7]
    multi_val_split = [0.2, 0.2, 0.80, 0.2, 0.2, 0.80]
    runs:int = len(multi_epochs)


    def Execute(self):
        if self.MULTI_RUN: return self.ExectuteMulti()
        else: return self.ExecuteSingle()

    def ExectuteMulti(self):
        for dataset in self.datasets:

            self.DATASET_NAME = dataset
            self.fname = dataset.split('/')[0]
            self.DATASET = './Datasets/Raw/'+dataset

            for run in range(self.runs):
                # Reset Logger
                sys.stdout = sys.__stdout__
                sys.stdout.flush()

                # Set network changes
                self.EPOCHS = self.multi_epochs[run]
                self.HOP_STEPS = self.multi_hops[run]
                self.VALIDATION_SPLIT = self.multi_val_split[run]

                #Set name elements and time elements
                self.TIME_NOW:str = strftime("%Y%m%d %H_%M_%S", gmtime())
                self.FOLDERNAME:str = "graph2seq_model_" + self.fname + "_DT_" + self.TIME_NOW + "/"
                self.MODEL_DESC:str = self.FOLDERNAME + "model_" + self.fname + "_eps_"+ str(self.EPOCHS) + "_HOPS_" + str(self.HOP_STEPS) + "_GVSize_" + str(self.GLOVE_VEC_SIZE) + "_DT_" + self.TIME_NOW + "_"
                self.ExecuteSingle()

    def ExecuteSingle(self):
        """
        The main method of the tool.
        It provides 2 functions:
            1. Storing the cleaned version of the passed AMR file
            2. Execute the network on the given dataset (includes cleaning but no storing of the AMR). 
        """  
        try:
            print("Model Folder Path:", self.FOLDERNAME)
            if not FolderCreator(self.FOLDERNAME).Create(): 
                print("Result folder was not being created!")
                return False

            sys.stdout = FACLogger(self.FOLDERNAME, self.fname + "_Log")
            self._available_gpus = KTFGPUHandler().GetAvailableGPUs()

            print("\n#######################################")
            print("######## Graph to Sequence ANN ########")
            print("#######################################\n")

            print("~~~~~~~~~~ System Informations ~~~~~~~~")
            print("Used OS:\t\t=> ", pf.system())
            print("Release:\t\t=> ", pf.release())
            print("Version:\t\t=> ", pf.version())
            print("Architecture:\t\t=> ", pf.architecture())
            print("Machine:\t\t=> ", pf.machine())
            print("Platform:\t\t=> ", pf.platform())
            print("CPU:\t\t\t=> ", pf.processor())
            print("GPUs:\t\t\t=> ", self._available_gpus)
            print("Python Version:\t\t=> ", pf.python_version())
            print("Tensorflow version: \t=> ", tf.__version__)
            print("Keras version: \t\t=> ", keras.__version__, '\n')

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.TF_CPP_MIN_LOG_LEVEL

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            if self.SAVING_CLEANED_AMR:
                self.ExecuteCleanedAMRStoring(  in_dataset=self.DATASET,
                                                in_extender=self.EXTENDER,
                                                in_max_length=self.MAX_LENGTH_DATA,
                                                is_show=self.SHOW_FEEDBACK,
                                                keep_edges=self.KEEP_EDGES,
                                                as_amr=True)
            else:
                self.ExecuteNetwork(in_dataset=self.DATASET, 
                                    in_glove=self.GLOVE, 
                                    in_extender=self.EXTENDER,
                                    in_max_length=self.MAX_LENGTH_DATA,
                                    in_vocab_size=self.GLOVE_VOCAB_SIZE,
                                    out_dim_emb=self.GLOVE_OUTPUT_DIM,
                                    is_show=self.SHOW_FEEDBACK,
                                    keep_edges=self.KEEP_EDGES)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.ExecuteTool]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def ExecuteNetwork(self, in_dataset, in_glove, in_extender="output", in_max_length=-1, in_vocab_size=20000, out_dim_emb=100, is_show=True, keep_edges=False):
        """
        This function executes the training pipeline for the graph2seq tool.

        This includes:
            1. Clean the AMR Dataset
            2. Convert to Lookup
            3. Initiate Glove Embedding
            4. Collect Worde and Sentence Features
            5. Construct Network Model
            6. Train Network Model
            7. Save Network Model Weights
            8. Plot Train Results

            :param in_dataset: input dataset
            :param in_glove: path to glove word vector file
            :param in_extender: output file entension
            :param in_max_length: max allowed length of sentences (max_len Semantic = 2x max_len sentences)
            :param in_vocab_size: the max amount of most occouring words you desire to store
            :param out_dim_emb: dimension for the glove embedding layer output dimension
            :param is_show: show process steps as console feedback
            :param keep_edges: keep edgesduring cleaning
        """   
        try:
            print("\n###### AMR Dataset Preprocessing ######")
            pipe = DatasetPipeline(in_path=in_dataset, 
                                output_path_extender=in_extender, 
                                max_length=in_max_length, 
                                show_feedback=is_show,
                                keep_edges=keep_edges,
                                min_cardinality=self.MIN_NODE_CARDINALITY, 
                                max_cardinality=self.MAX_NODE_CARDINALITY)

            datapairs = pipe.ProvideData()
            max_cardinality = pipe._max_observed_nodes_cardinality
            self._dataset_size = len(datapairs)
            self._predict_split_value = self.TestSplitSize(self._dataset_size, self._predict_percentage_split)
            print('Found Datapairs:\n\t=> [', self._dataset_size, '] for allowed graph node cardinality interval [',self.MIN_NODE_CARDINALITY,'|',self.MAX_NODE_CARDINALITY,']')
            pipe.PlotCardinalities(self.MODEL_DESC)
            self.DatasetLookUpEqualization(datapairs, max_cardinality)
                
            max_sequence_len:int = (max_cardinality * 2) if max_cardinality != pipe._max_words_sentences else -1
                
            print("~~~~~~ Example Target Data Pipe ~~~~~~~")
            print("Target 0", datapairs[0][0])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            print("#######################################\n")
            print("########## Dataset Tokenizer ##########")

            glove_dataset_processor = GloVeDatasetPreprocessor( nodes_context=datapairs, 
                                                                vocab_size=in_vocab_size,
                                                                max_sequence_length=max_sequence_len,
                                                                show_feedback=True)
            _, _, edge_fw_look_up, edge_bw_look_up, vectorized_inputs, vectorized_targets, dataset_nodes_values, _ = glove_dataset_processor.Execute()

            print("~~~~~~ Example Target Tokenizer ~~~~~~~")
            #glove_dataset_processor.Convert("Input 0", glove_dataset_processor.tokenizer, vectorized_inputs[0])
            glove_dataset_processor.Convert("Target 0", glove_dataset_processor.tokenizer, vectorized_targets[0])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            print("#######################################\n")
            print("######## Glove Embedding Layer ########")
            glove_embedding = GloVeEmbedding(   max_cardinality=max_cardinality, 
                                                vocab_size=in_vocab_size, 
                                                tokenizer=glove_dataset_processor,
                                                max_sequence_length= max_sequence_len,
                                                glove_file_path=self.GLOVE, 
                                                output_dim=out_dim_emb, 
                                                batch_size=self.BATCH_SIZE,
                                                show_feedback=True)

            datasets_nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(dataset_nodes_values)
            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer(True)

            print("Reminder: [1 ----> <go>] and [2 ----> <eos>]")
           

            #vectorized_inputs = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_inputs, check_cardinality=False)
            #vectorized_targets = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_targets, check_cardinality=False)

            vectorized_inputs = np.expand_dims(vectorized_inputs, axis=-1)
            #vectorized_targets = np.expand_dims(vectorized_targets, axis=-1)

            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()

            print('Embedding Resources:\n\t => Free (in further steps unused) resources!', )
            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()

            print("#######################################\n")
            print("#### Prepare Train and Predictset #####")

            #TODO: Missing Teacherforcing -> each graph repeat for each word in sentence!
            #TODO: Call -> GenerateDatasetTeacherForcing

            train_x, train_y, test_x, test_y = self.GenerateDatasetTeacherForcing(  split_border = (self._dataset_size - self._predict_split_value), 
                                                                                    nodes_embedding = datasets_nodes_embedding, 
                                                                                    fw_look_up = edge_fw_look_up, 
                                                                                    bw_look_up = edge_bw_look_up, 
                                                                                    vecs_input_sentences = vectorized_inputs,
                                                                                    vecs_target_sentences = vectorized_targets)
            '''
            train_x = [ datasets_nodes_embedding[:self._dataset_size - self._predict_split_value], 
                        edge_fw_look_up[:self._dataset_size - self._predict_split_value], 
                        edge_bw_look_up[:self._dataset_size - self._predict_split_value],
                        vectorized_inputs[:self._dataset_size - self._predict_split_value]]

            test_x = [  datasets_nodes_embedding[self._dataset_size - self._predict_split_value:], 
                        edge_fw_look_up[self._dataset_size - self._predict_split_value:], 
                        edge_bw_look_up[self._dataset_size - self._predict_split_value:],
                        vectorized_inputs[self._dataset_size - self._predict_split_value:]]

            train_y = vectorized_targets[:self._dataset_size - self._predict_split_value]
            test_y = vectorized_targets[self._dataset_size - self._predict_split_value:]
            '''
            print("#######################################\n")
            print("########## Construct Network ##########")

            builder = ModelBuilder( input_enc_dim=self.GLOVE_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=vectorized_targets.shape[2],
                                    batch_size=self.BATCH_SIZE)

            print("Builder!")

            encoder, graph_embedding_encoder_states = builder.BuildGraphEmbeddingEncoder(hops=self.HOP_STEPS)

            print("Encoder!")

            model = builder.BuildGraphEmbeddingDecoder( embedding=glove_embedding_layer(builder.get_decoder_inputs()), 
                                                        encoder=encoder,
                                                        act=activations.softmax,
                                                        prev_memory_state=graph_embedding_encoder_states[0],  
                                                        prev_carry_state=graph_embedding_encoder_states[1])

            print("Decoder!")

            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model, metrics=self._accurracy, loss = 'categorical_crossentropy')
            builder.Plot(model=model, file_name=self.MODEL_DESC+'model_graph.png')

            print("#######################################\n")
            print("########### Starts Training ###########")

            base_lr = BaseLogger()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=self.VERBOSE)

            history = model.fit(train_x, 
                                train_y,
                                batch_size = self.BATCH_SIZE,
                                epochs=self.EPOCHS, 
                                verbose=self.VERBOSE, 
                                shuffle=self.SHUFFLE_DATASET,
                                validation_split=self.VALIDATION_SPLIT,
                                callbacks=[base_lr, reduce_lr])

            print("History Keys: ", list(history.history.keys()))

            print("#######################################\n")
            print("######## Plot Training Results ########")

            plotter = HistoryPlotter(   model_description = self.MODEL_DESC, 
                                        path = None, 
                                        history = history,
                                        new_style = False)

            plotter.PlotHistory()           
            print("#######################################\n")
            print("########### Predict  Results ##########")

            y_pred = model.predict(test_x, batch_size = self.BATCH_SIZE, verbose=0)

            correct:int = 0
            for i in range(len(y_pred)):
                print("Test: ", test_y[i])
                print("Pred: ",  y_pred[i])
                if array_equal(test_y[i], y_pred[i]):
                    correct += 1
            print('Prediction Accuracy: %.2f%%' % (float(correct)/float(self._predict_split_value)*100.0))

            print("#######################################\n")
            print("######## Process End ########")

        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.ExecuteNetwork]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def ExecuteCleanedAMRStoring(self, 
                             in_dataset:str, 
                             in_extender:str, 
                             in_max_length:int, 
                             is_show:bool, 
                             keep_edges:bool, 
                             as_amr:bool):
        """
        This function stores the cleaned amr dataset, only!
            :param in_dataset:str: input dataset
            :param in_extender:str: output file entension
            :param in_max_length:int: max allowed length of sentences (max_len Semantic = 2x max_len sentences)
            :param is_show:bool: show process steps as console feedback
            :param keep_edges:bool: keep edgesduring cleaning
            :param as_amr:bool: save as amr if true else as json
        """   
        try:
            pipe = DatasetPipeline(in_path=in_dataset, 
                            output_path_extender=in_extender, 
                            max_length=in_max_length, 
                            show_feedback=is_show, 
                            keep_edges=keep_edges,
                            min_cardinality=self.MIN_NODE_CARDINALITY, 
                            max_cardinality=self.MAX_NODE_CARDINALITY
                            )

            pipe.SaveData(as_amr=as_amr)
            print('Finished storing process!')
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.ExecuteCleanedAMRStoring]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def SingleLookUpEqualization(self, datapair:list, max_card:int):
        """
        This function wraps the MatrixExpansionWithZeros function for the foward and backward edge look up for a datapair.
            :param datapair:list: single elemt of the DatasetPipeline result 
            :param max_card:int: desired max cardinality 
        """
        try:
            assert (datapair[1][0] is not None), ('Wrong input for dataset edge look up size equalization!')
            elem1 = MatrixExpansionWithZeros(datapair[1][0][0], max_card)
            elem2 = MatrixExpansionWithZeros(datapair[1][0][1], max_card)
            assert (elem1.shape == elem2.shape and elem1.shape == (max_card,max_card)), ("Results have wrong shape!")
            return [elem1,elem2]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.SingleLookUpEqualization]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def DatasetLookUpEqualization(self, datapairs:list ,max_cardinality:int):
        """
        This function equalizes all datasets neighbourhood look up matrices to a given max cardinality.
            :param datapairs:list: the dataset
            :param max_cardinality:int: the given cardinality
        """   
        try:
            assert (max_cardinality > 0), ("Max graph nodes cardinality was 0!")
            for datapair in datapairs:
                datapair[1][0] = self.SingleLookUpEqualization(datapair, max_cardinality)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.DatasetLookUpEqualization]. Arguments:\n{1!r}"
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
                plt.savefig((self.MODEL_DESC+'plot.'+image_type), orientation=orientation)
            else: 
                plt.savefig((self.MODEL_DESC+extender+'.'+image_type), orientation=orientation)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.SavePyPlotToFile]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def TestSplitSize(self, dataset_size:int, split_percentage:float):
        """
        This method return a number for the size of desired test samples from dataset by a given percentage.
            :param dataset_size:int: size of the whole datset
            :param split_percentage:float: desired test size percentage from dataset
        """   
        try:
            if isNumber(dataset_size) and isNumber(split_percentage):
                return round((dataset_size * split_percentage)/100.0)
            else:
                return -1
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.SplitData]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GenerateDatasetTeacherForcing(self, split_border:int, nodes_embedding:np.ndarray, fw_look_up:np.ndarray, bw_look_up:np.ndarray, vecs_input_sentences:np.ndarray, vecs_target_sentences:np.ndarray):

        try:
            if len(nodes_embedding) == len(fw_look_up) == len(bw_look_up) == len(vecs_input_sentences) == len(vecs_target_sentences):                
                path_partial:str = (self.FOLDERNAME + self.fname + self.fname + "_DT_" + self.TIME_NOW + "_")

                train_x = [] 
                train_y = []
                test_x = []
                test_y = []

                nodes_emb = None
                forward_look_up = None
                backward_look_up = None
                vecs_input_words = None
                vecs_target_words = None

                print("Start Dataset Generator\n[", end = '')

                for s_idx in range(len(vecs_input_sentences)):

                    tmp_nodes_emb = nodes_embedding[s_idx]
                    tmp_forward_look_up = fw_look_up[s_idx]
                    tmp_backward_look_up = bw_look_up[s_idx]
                    tmp_vecs_input_words = np.trim_zeros(vecs_input_sentences[s_idx])
                    tmp_vecs_target_words = np.trim_zeros(vecs_target_sentences[s_idx])
                    qualified_entries = np.count_nonzero(tmp_vecs_input_words) - 1      #The -1 mean i will not include the the first copy since we keep the initial as well!
                    
                    if (nodes_emb is None) and (forward_look_up is None) and (backward_look_up is None) and (vecs_input_words is None) and (vecs_target_words is None):
                        nodes_emb = RepeatNTimsNdArray(times=qualified_entries, array=tmp_nodes_emb)
                        forward_look_up = RepeatNTimsNdArray(times=qualified_entries, array=tmp_forward_look_up)
                        backward_look_up = RepeatNTimsNdArray(times=qualified_entries, array=tmp_backward_look_up)
                        vecs_input_words = tmp_vecs_input_words.reshape((tmp_vecs_input_words.shape[0],))[:-1]
                        vecs_target_words = tmp_vecs_target_words.reshape((tmp_vecs_target_words.shape[0],))

                    else:
                        nodes_emb = ConcatenateNdArray(nodes_emb, RepeatNTimsNdArray(times=qualified_entries, array=tmp_nodes_emb))
                        forward_look_up = ConcatenateNdArray(forward_look_up, RepeatNTimsNdArray(times=qualified_entries, array=tmp_forward_look_up))
                        backward_look_up = ConcatenateNdArray(backward_look_up, RepeatNTimsNdArray(times=qualified_entries, array=tmp_backward_look_up))
                        vecs_input_words = ConcatenateNdArray(vecs_input_words, tmp_vecs_input_words.reshape((tmp_vecs_input_words.shape[0],))[:-1])
                        vecs_target_words = ConcatenateNdArray(vecs_target_words, tmp_vecs_target_words.reshape((tmp_vecs_target_words.shape[0],)))
                
                    if ((s_idx+1)%250 is 0 ): 
                        print(" >", end = '')
                    if ((s_idx+1) == len(vecs_input_sentences)): 
                        print(" ] Done!")
                
                np.savetxt((path_partial + "nodes_emb.out"), nodes_emb)
                np.savetxt((path_partial + "forward_look_up.out"), forward_look_up)
                np.savetxt((path_partial + "backward_look_up.out"), backward_look_up)
                np.savetxt((path_partial + "vecs_input_words.out"), vecs_input_words)
                np.savetxt((path_partial + "vecs_target_words.out"), vecs_target_words)

                sys.exit(0)
                
            else:
                assert not(len(nodes_embedding) == len(fw_look_up) == len(bw_look_up) == len(vecs_input_sentences) == len(vecs_target_sentences)), "The given inputs of GenerateDatasetTeacherForcing aren't machting at first dimension!"
                sys.exit(0)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.GenerateDatasetTeacherForcing]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)



if __name__ == "__main__":
    Graph2SeqInKeras().Execute()