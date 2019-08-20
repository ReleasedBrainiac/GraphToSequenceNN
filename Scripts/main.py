import os
import sys
import platform as pf
import numpy as np
from numpy import array_equal, argmax
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import History, ReduceLROnPlateau, BaseLogger, EarlyStopping, ModelCheckpoint

from time import gmtime, strftime
from Logger.Logger import FACLogger, FolderCreator
from Configurable.ProjectConstants  import Constants
from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import DatasetSplitIndex, isNone
from GloVeHandler.GloVeDatasetPreprocessor import GloVeDatasetPreprocessor
from GloVeHandler.GloVeEmbedding import GloVeEmbedding
from DatasetHandler.FileWriter import Writer
from NetworkHandler.Builder.ModelBuilder import ModelBuilder
from Plotter.PlotHistory import HistoryPlotter
from NetworkHandler.TensorflowSetup.UsageHandlerGPU import KTFGPUHandler
from DatasetHandler.NumpyHandler import NumpyDatasetHandler, NumpyDatasetPreprocessor
from GraphHandler.SemanticMatrixBuilder import MatrixHandler

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
    EPOCHS:int = 10
    VERBOSE:int = 1
    BATCH_SIZE:int = 1
    DATASET_NAME:str = 'AMR Bio/amr-release-training-bio.txt' #'Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt' #
    fname = DATASET_NAME.split('/')[0]
    DATASET:str = './Datasets/Raw/'+DATASET_NAME
    GLOVE:str = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    GLOVE_VEC_SIZE:int = 100
    PLOT:str = "plot.png"
    EXTENDER:str = "amr.cleaner.ouput"
    MAX_LENGTH_DATA:int = -1
    SHOW_GLOBAL_FEEDBACK:bool = False
    SAVE_PLOTS = True
    KEEP_EDGES:bool = True
    GLOVE_OUTPUT_DIM:int = 100
    GLOVE_VOCAB_SIZE:int = 5000
    VALIDATION_SPLIT:float = 0.8
    MIN_NODE_CARDINALITY:int = 15
    MAX_NODE_CARDINALITY:int = 35
    HOP_STEPS:int = 3
    SHUFFLE_DATASET:bool = True

    USE_PREPARED_DATASET:bool = False
    PREPARED_DS_PATH:str = 'graph2seq_model_AMR Bio_DT_20190808 09_23_25/AMR BioAMR Bio_DT_20190808 09_23_25'

    _accurracy:list = ['acc']
    _loss_function:str = 'sparse_categorical_crossentropy'
    _last_activation:str = 'softmax'
    _available_gpus = None
    _predict_percentage_split:float = 8.0
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
    multi_val_split = [0.75, 0.65, 0.80, 0.75, 0.65, 0.80]
    runs:int = len(multi_epochs)


    def Execute(self):
        if self.MULTI_RUN: 
            return self.ExectuteMulti()
        else: 
            #return self.ExecuteSingle()
            return self.ToolPipe()

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
                #self.ExecuteSingle()
                self.ToolPipe()

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
            self.ExecuteNetwork(in_dataset=self.DATASET,
                                in_extender=self.EXTENDER,
                                in_max_length=self.MAX_LENGTH_DATA,
                                in_vocab_size=self.GLOVE_VOCAB_SIZE,
                                out_dim_emb=self.GLOVE_OUTPUT_DIM,
                                is_show=self.SHOW_GLOBAL_FEEDBACK,
                                keep_edges=self.KEEP_EDGES)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.ExecuteTool]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def SystemInfo(self):
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
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.SystemInfo]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DatasetPreprocessor(self, in_dataset:str, in_extender:str="output", in_max_length:int=-1, show_processor_feedback:bool=False, keep_edges:bool=True, semantic_amr_string:bool=False):
        try:
            pipe = DatasetPipeline( in_path=in_dataset, 
                                    output_path_extender=in_extender, 
                                    max_length=in_max_length, 
                                    show_feedback=show_processor_feedback,
                                    keep_edges=keep_edges,
                                    min_cardinality=self.MIN_NODE_CARDINALITY, 
                                    max_cardinality=self.MAX_NODE_CARDINALITY,
                                    stringified_amr=semantic_amr_string)

            datapairs = pipe.ProvideData()
            max_cardinality = pipe._max_observed_nodes_cardinality
            self._dataset_size = len(datapairs)
            self._predict_split_value = DatasetSplitIndex(self._dataset_size, self._predict_percentage_split)

            pipe.PlotCardinalities(self.MODEL_DESC)
            if self.SHOW_GLOBAL_FEEDBACK:
                print('Found Datapairs:\n\t=> [', self._dataset_size, '] for allowed graph node cardinality interval [',self.MIN_NODE_CARDINALITY,'|',self.MAX_NODE_CARDINALITY,']')

            MatrixHandler().DatasetLookUpEqualization(datapairs, max_cardinality)
            max_sequence_len:int = (max_cardinality * 2) if max_cardinality != pipe._max_words_sentences else -1

            if self.SHOW_GLOBAL_FEEDBACK:
                print("~~~~~~ Example Target Data Pipe ~~~~~~~")
                print("Target 0", datapairs[0][0])
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            return [max_sequence_len, max_cardinality, datapairs]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.DatasetPreprocessor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def GlovePreprocessor(self, datapairs:list, in_vocab_size:int, max_sequence_len:int=-1, show_processor_feedback:bool=True):
        try:
            print("#######################################\n")
            print("########## Dataset Tokenizer ##########")

            glove_dataset_processor = GloVeDatasetPreprocessor( nodes_context=datapairs, 
                                                                vocab_size=in_vocab_size,
                                                                max_sequence_length=max_sequence_len,
                                                                show_feedback=show_processor_feedback)
            _, _, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, dataset_nodes_values, _ = glove_dataset_processor.Execute()

            if self.SHOW_GLOBAL_FEEDBACK:
                print("~~~~~~ Example Target Tokenizer ~~~~~~~")
                glove_dataset_processor.Convert("Input 0", glove_dataset_processor.tokenizer, vectorized_inputs[0])
                glove_dataset_processor.Convert("Target 0", glove_dataset_processor.tokenizer, vectorized_targets[0])
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            return [glove_dataset_processor, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, dataset_nodes_values]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.GlovePreprocessor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)            

    def GloveEmbedding(self, glove_dataset_processor:GloVeDatasetPreprocessor, dataset_nodes_values:list, vectorized_inputs:list, vectorized_targets:list, max_cardinality:int, max_sequence_len:int, in_vocab_size:int, out_dim_emb:int, show_processor_feedback:bool=True, embedding_input_wordwise:bool=True, nodes_to_embedding:bool=False):
        try:
            print("#######################################\n")
            print("######## Glove Embedding Layer ########")
            glove_embedding = GloVeEmbedding(   max_cardinality=max_cardinality, 
                                                vocab_size=in_vocab_size, 
                                                tokenizer=glove_dataset_processor,
                                                max_sequence_length= max_sequence_len,
                                                glove_file_path=self.GLOVE, 
                                                output_dim=out_dim_emb, 
                                                batch_size=self.BATCH_SIZE,
                                                show_feedback=show_processor_feedback)

            nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(dataset_nodes_values)
            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer(embedding_input_wordwise)

            if self.SHOW_GLOBAL_FEEDBACK: 
                print("Reminder: [1 ----> <go>] and [2 ----> <eos>]")
           

            if nodes_to_embedding:
                vectorized_inputs = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_inputs, check_cardinality=False)
                vectorized_targets = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_targets, check_cardinality=False)
            else:
                vectorized_inputs = np.expand_dims(vectorized_inputs, axis=-1)
                vectorized_targets = np.expand_dims(vectorized_targets, axis=-1)

            return [nodes_embedding, vectorized_inputs, vectorized_targets, glove_embedding_layer]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.GloveEmbedding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def DatasetConvertToTeacherForcing(self, nodes_embedding:list, fw_look_up:list, bw_look_up:list, vectorized_inputs:list, vectorized_targets:list, save_dataset:bool=False):
        try:
            print("#######################################\n")
            print("### Create Word Wise Teacherforcing ###")


            generator = NumpyDatasetPreprocessor(folder_path=self.FOLDERNAME)
            if not self.USE_PREPARED_DATASET:
                nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = generator.PreprocessTeacherForcingDS(  nodes_embedding, 
                                                                                                                                        fw_look_up, 
                                                                                                                                        bw_look_up, 
                                                                                                                                        vectorized_inputs,
                                                                                                                                        vectorized_targets,
                                                                                                                                        save=save_dataset)
            else:
                nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = NumpyDatasetHandler(path=self.PREPARED_DS_PATH).LoadTeacherForcingDS()
            

            return [generator, nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.DatasetConvertToTeacherForcing]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)   

    def NetworkInputPreparation(self, generator:NumpyDatasetPreprocessor, nodes_embedding:list, fw_look_up, bw_look_up:list, vectorized_inputs:list, vectorized_targets:list):
        try:
            print("#######################################\n")
            print("#### Prepare Train and Predictset #####")
            if isNone(generator):
                generator = NumpyDatasetPreprocessor(None)

            train_x, train_y, test_x, test_y = generator.NetworkInputPreparation(   nodes_embedding, 
                                                                                    fw_look_up, 
                                                                                    bw_look_up, 
                                                                                    vectorized_inputs,
                                                                                    vectorized_targets,
                                                                                    (self._dataset_size - self._predict_split_value))

            return [train_x, train_y, test_x, test_y]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkInputPreparation]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex) 

    def NetworkConstruction(self, target_shape:tuple, max_cardinality:int, glove_embedding_layer:keras.layers.embeddings.Embedding):
        try:
            print("#######################################\n")
            print("####### Construct Network Model #######")

            input_dec_dim = 1 if len(target_shape) <= 1 else target_shape[-1]

            print("Builder Init!")

            builder = ModelBuilder( input_enc_dim=self.GLOVE_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=input_dec_dim,
                                    batch_size=self.BATCH_SIZE)

            print("Build Encoder!")

            encoder, graph_embedding_encoder_states = builder.BuildGraphEmbeddingEncoder(hops=self.HOP_STEPS)

            print("Build Decoder!")

            model = builder.BuildGraphEmbeddingDecoder( embedding=glove_embedding_layer(builder.get_decoder_inputs()), 
                                                        encoder=encoder,
                                                        act=self._last_activation,
                                                        prev_memory_state=graph_embedding_encoder_states[0],  
                                                        prev_carry_state=graph_embedding_encoder_states[1])

            print("Build Finalize and Plot!")
            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model, metrics=self._accurracy, loss = self._loss_function)
            builder.Plot(model=model, file_name=self.MODEL_DESC+'model_graph.png')

            return model
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkConstruction]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def NetworkTrain(self, model:keras.engine.training.Model, train_x:list, train_y:np.ndarray):
        try:
            print("#######################################\n")
            print("########### Starts Training ###########")

            base_lr = BaseLogger()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=self.VERBOSE)
            es = EarlyStopping(monitor='val_loss', mode='min', patience=100)
            mc = ModelCheckpoint(self.MODEL_DESC+'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


            return model.fit(   train_x, 
                                train_y,
                                batch_size = self.BATCH_SIZE,
                                epochs=self.EPOCHS, 
                                verbose=self.VERBOSE, 
                                shuffle=self.SHUFFLE_DATASET,
                                validation_split=self.VALIDATION_SPLIT,
                                callbacks=[base_lr, reduce_lr, es, mc])
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkTrain]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def NetworkPlotResults(self, history, new_style:bool = False):
        try:
            print("#######################################\n")
            print("######## Plot Training Results ########")

            print(type(history))
            if self.SHOW_GLOBAL_FEEDBACK:
                print("History Keys: ", list(history.history.keys()))

            plotter = HistoryPlotter(   model_description = self.MODEL_DESC, 
                                        path = None, 
                                        history = history,
                                        new_style = new_style)

            plotter.PlotHistory()
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkPlotResults]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  
    
    def NetworkPredict(self, model, test_x, test_y):
        try:
            print("#######################################\n")
            print("########### Predict  Results ##########")

            print(type(model))

            y_pred = model.predict(test_x, batch_size = self.BATCH_SIZE, verbose=self.VERBOSE)

            correct:int = 0
            for i in range(len(y_pred)):
                print("Test: ", test_y[i])
                print("Pred: ",  y_pred[i])
                if array_equal(test_y[i], y_pred[i]):
                    correct += 1
            print('Prediction Accuracy: %.2f%%' % (float(correct)/float(self._predict_split_value)*100.0))
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkPredict]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def ToolPipe(self):
        try:
            self.SystemInfo()

            max_sequence_len, max_cardinality, datapairs = self.DatasetPreprocessor(in_dataset=self.DATASET,
                                                                                    in_extender=self.EXTENDER,
                                                                                    in_max_length=self.MAX_LENGTH_DATA, 
                                                                                    keep_edges=self.KEEP_EDGES)
            
            glove_dataset_processor, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, dataset_nodes_values = self.GlovePreprocessor(  datapairs=datapairs, 
                                                                                                                                                    in_vocab_size=self.GLOVE_VOCAB_SIZE, 
                                                                                                                                                    max_sequence_len=max_sequence_len)

            nodes_embedding, vectorized_inputs, vectorized_targets, glove_embedding_layer = self.GloveEmbedding(glove_dataset_processor=glove_dataset_processor, 
                                                                                                                dataset_nodes_values=dataset_nodes_values, 
                                                                                                                vectorized_inputs=vectorized_inputs, 
                                                                                                                vectorized_targets=vectorized_targets, 
                                                                                                                max_cardinality=max_cardinality, 
                                                                                                                max_sequence_len=max_sequence_len, 
                                                                                                                in_vocab_size=self.GLOVE_VOCAB_SIZE, 
                                                                                                                out_dim_emb=self.GLOVE_OUTPUT_DIM, 
                                                                                                                show_processor_feedback=True, 
                                                                                                                embedding_input_wordwise=True, 
                                                                                                                nodes_to_embedding=False)

            generator, nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = self.DatasetConvertToTeacherForcing(nodes_embedding=nodes_embedding, 
                                                                                                                                            fw_look_up=fw_look_up, 
                                                                                                                                            bw_look_up=bw_look_up, 
                                                                                                                                            vectorized_inputs=vectorized_inputs, 
                                                                                                                                            vectorized_targets=vectorized_targets, 
                                                                                                                                            save_dataset=False)

            train_x, train_y, test_x, test_y = self.NetworkInputPreparation(generator=generator, 
                                                                            nodes_embedding=nodes_embedding, 
                                                                            fw_look_up=fw_look_up, 
                                                                            bw_look_up=bw_look_up, 
                                                                            vectorized_inputs=vectorized_inputs, 
                                                                            vectorized_targets=vectorized_targets)

            model = self.NetworkConstruction(   target_shape=vectorized_targets.shape, 
                                                max_cardinality=max_cardinality, 
                                                glove_embedding_layer=glove_embedding_layer)

            history = self.NetworkTrain(model, train_x, train_y)

            self.NetworkPlotResults(history)

            self.NetworkPredict(model, test_x, test_y)
            print("#######################################\n")
            print("######## Process End ########")
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.ToolPipe]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)     
    



    def ExecuteNetwork(self, in_dataset:str, in_extender:str="output", in_max_length:int=-1, in_vocab_size:int=20000, out_dim_emb:int=100, is_show:bool=True, keep_edges:bool=False, semantic_amr_string:bool=False):
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

            :param in_dataset:str: input dataset
            :param in_extender:str: output file entension
            :param in_max_length:int: max allowed length of sentences (max_len Semantic = 2x max_len sentences)
            :param in_vocab_size:int: the max amount of most occouring words you desire to store
            :param out_dim_emb:int: dimension for the glove embedding layer output dimension
            :param is_show:bool: show process steps as console feedback
            :param keep_edges:bool: keep edgesduring cleaning
            :param semantic_amr_string:bool: semantic element as string not as matrices graph lookup
        """   
        try:

            print("\n###### AMR Dataset Preprocessing ######")
            pipe = DatasetPipeline(in_path=in_dataset, 
                                output_path_extender=in_extender, 
                                max_length=in_max_length, 
                                show_feedback=is_show,
                                keep_edges=keep_edges,
                                min_cardinality=self.MIN_NODE_CARDINALITY, 
                                max_cardinality=self.MAX_NODE_CARDINALITY,
                                stringified_amr=semantic_amr_string)


            datapairs = pipe.ProvideData()
            max_cardinality = pipe._max_observed_nodes_cardinality
            self._dataset_size = len(datapairs)
            self._predict_split_value = DatasetSplitIndex(self._dataset_size, self._predict_percentage_split)
            print('Found Datapairs:\n\t=> [', self._dataset_size, '] for allowed graph node cardinality interval [',self.MIN_NODE_CARDINALITY,'|',self.MAX_NODE_CARDINALITY,']')

            pipe.PlotCardinalities(self.MODEL_DESC)
            MatrixHandler().DatasetLookUpEqualization(datapairs, max_cardinality)
                
            max_sequence_len:int = (max_cardinality * 2) if max_cardinality != pipe._max_words_sentences else -1
                
            ###########################################################################################################################################################################

            print("~~~~~~ Example Target Data Pipe ~~~~~~~")
            print("Target 0", datapairs[0][0])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            print("#######################################\n")
            print("########## Dataset Tokenizer ##########")

            glove_dataset_processor = GloVeDatasetPreprocessor( nodes_context=datapairs, 
                                                                vocab_size=in_vocab_size,
                                                                max_sequence_length=max_sequence_len,
                                                                show_feedback=True)
            _, _, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, dataset_nodes_values, _ = glove_dataset_processor.Execute()

            print("~~~~~~ Example Target Tokenizer ~~~~~~~")
            #glove_dataset_processor.Convert("Input 0", glove_dataset_processor.tokenizer, vectorized_inputs[0])
            glove_dataset_processor.Convert("Target 0", glove_dataset_processor.tokenizer, vectorized_targets[0])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            ###########################################################################################################################################################################

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

            nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(dataset_nodes_values)
            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer(True)

            print("Reminder: [1 ----> <go>] and [2 ----> <eos>]")
           

            #vectorized_inputs = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_inputs, check_cardinality=False)
            #vectorized_targets = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_targets, check_cardinality=False)

            vectorized_inputs = np.expand_dims(vectorized_inputs, axis=-1)
            vectorized_targets = np.expand_dims(vectorized_targets, axis=-1)

            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()

            print('Embedding Resources:\n\t => Free (in further steps unused) resources!', )
            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()

            ###########################################################################################################################################################################

            print("#######################################\n")
            print("#### Prepare Train and Predictset #####")


            generator = NumpyDatasetPreprocessor(folder_path = self.FOLDERNAME)
            if not self.USE_PREPARED_DATASET:
                nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = generator.PreprocessTeacherForcingDS(  nodes_embedding, 
                                                                                                                                        fw_look_up, 
                                                                                                                                        bw_look_up, 
                                                                                                                                        vectorized_inputs,
                                                                                                                                        vectorized_targets,
                                                                                                                                        save=False)
            else:
                nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = NumpyDatasetHandler(path=self.PREPARED_DS_PATH).LoadTeacherForcingDS()
            

            split_border = (self._dataset_size - self._predict_split_value)
            train_x, train_y, test_x, test_y = generator.NetworkInputPreparation(   nodes_embedding, 
                                                                                    fw_look_up, 
                                                                                    bw_look_up, 
                                                                                    vectorized_inputs,
                                                                                    vectorized_targets,
                                                                                    split_border)

            ###########################################################################################################################################################################                                              
           
            print("#######################################\n")
            print("########## Construct Network ##########")

            input_dec_dim = 1 if len(vectorized_targets.shape) <= 1 else vectorized_targets.shape[-1]

            builder = ModelBuilder( input_enc_dim=self.GLOVE_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=input_dec_dim,
                                    batch_size=self.BATCH_SIZE)

            print("Builder Start!")

            encoder, graph_embedding_encoder_states = builder.BuildGraphEmbeddingEncoder(hops=self.HOP_STEPS)

            print("Encoder Constructed!")

            model = builder.BuildGraphEmbeddingDecoder( embedding=glove_embedding_layer(builder.get_decoder_inputs()), 
                                                        encoder=encoder,
                                                        act='relu',
                                                        prev_memory_state=graph_embedding_encoder_states[0],  
                                                        prev_carry_state=graph_embedding_encoder_states[1])

            print("Decoder Constructed!")

            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model, metrics=self._accurracy, loss = 'logcosh')
            builder.Plot(model=model, file_name=self.MODEL_DESC+'model_graph.png')

            ###########################################################################################################################################################################

            print("#######################################\n")
            print("########### Starts Training ###########")

            base_lr = BaseLogger()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=self.VERBOSE)
            es = EarlyStopping(monitor='val_loss', mode='min', patience=100)
            mc = ModelCheckpoint(self.MODEL_DESC+'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


            history = model.fit(train_x, 
                                train_y,
                                batch_size = self.BATCH_SIZE,
                                epochs=self.EPOCHS, 
                                verbose=self.VERBOSE, 
                                shuffle=self.SHUFFLE_DATASET,
                                validation_split=self.VALIDATION_SPLIT,
                                callbacks=[base_lr, reduce_lr, es, mc])

            ###########################################################################################################################################################################

            print("#######################################\n")
            print("######## Plot Training Results ########")

            print("History Keys: ", list(history.history.keys()))

            plotter = HistoryPlotter(   model_description = self.MODEL_DESC, 
                                        path = None, 
                                        history = history,
                                        new_style = False)

            plotter.PlotHistory()

            ###########################################################################################################################################################################

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

    def TestNewSetup(self):
        '''EPOCHS = 10

        for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))'''

if __name__ == "__main__":
    Graph2SeqInKeras().Execute()