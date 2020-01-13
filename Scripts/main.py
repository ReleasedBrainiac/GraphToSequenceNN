import os
import sys
import platform as pf
import numpy as np
from numpy import array_equal, argmax
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import History, ReduceLROnPlateau, BaseLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding

from time import gmtime, strftime
from Logger.Logger import FACLogger, FolderCreator
from Configurable.ProjectConstants  import Constants
from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import DatasetSplitIndex, isNone, isList
from GloVeHandler.GloVeDatasetPreprocessor import GloVeDatasetPreprocessor
from GloVeHandler.GloVeEmbedding import GloVeEmbedding
from DatasetHandler.FileWriter import Writer
from NetworkHandler.Builder.ModelBuilder import ModelBuilder
from Plotter.PlotHistory import HistoryPlotter
from NetworkHandler.TensorflowSetup.UsageHandlerGPU import KTFGPUHandler
from DatasetHandler.NumpyHandler import NumpyDatasetHandler, NumpyDatasetPreprocessor
from GraphHandler.SemanticMatrixBuilder import MatrixHandler
from keras.utils import to_categorical


#TODO bset tutorial => https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
#TODO K-Fold Keras => https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#TODO many-to-many => https://github.com/keras-team/keras/issues/1029
#TODO another resource => https://data-science-blog.com/blog/2017/12/20/maschinelles-lernen-klassifikation-vs-regression/
#TODO  IN MA => erläuterung => https://www.tensorflow.org/tutorials/structured_data/feature_columns
#TODO IN MA => Code Next Level => https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py?source=post_page
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
    #System
    TF_CPP_MIN_LOG_LEVEL:str = '2'
    MULTI_RUN:bool = False
    CPUS:int = os.cpu_count()
    GPUS = None #TODO: Bug on no old TF versions: KTFGPUHandler().GetAvailableGPUsTF2()

    #Logging
    SHOW_GLOBAL_FEEDBACK:bool = True
    TIME_NOW:str = strftime("%Y%m%d %H_%M_%S", gmtime())

    #Network
    EPOCHS:int = 15
    VERBOSE:int = 1
    VALIDATION_SPLIT:float = 0.2 # percentage of used samples from train set for cross validation ~> 0.2 = 20% for validation
    BATCH_SIZE:int = 8
    HOP_STEPS:int = 3
    WORD_WISE:bool = False
    USE_GLOVE:bool = False
    EMBEDDING_OUTPUT_DIM:int = 100

    #GLOVE
    GLOVE:str = './Datasets/GloVeWordVectors/glove.6B/glove.6B.'+str(EMBEDDING_OUTPUT_DIM)+'d.txt'
    GLOVE_VOCAB_SIZE:int = 15000

    #Dataset
    PREDICT_SPLIT:float = 0.2 # percentage of used samples form raw dataset for prediction ~> 0.2 = 20% for prediction 
    DATASET_NAME:str =    'Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt' #'AMR Bio/amr-release-training-bio.txt' #'2mAMR/2m.json'
    _fname = DATASET_NAME.split('/')[0]
    DATASET:str = './Datasets/Raw/'+DATASET_NAME
    EXTENDER:str = "amr.cleaner.ouput"
    MAX_LENGTH_DATA:int = -1
    KEEP_EDGES:bool = True
    MIN_NODE_CARDINALITY:int = 3
    MAX_NODE_CARDINALITY:int = 35
    USE_PREPARED_DATASET:bool = False
    PREPARED_DS_PATH:str = 'graph2seq_model_AMR Bio_DT_20190808 09_23_25/AMR BioAMR Bio_DT_20190808 09_23_25'
    SHUFFLE_DATASET:bool = True
    FOLDERNAME:str = "graph2seq_model_" + _fname + "_DT_" + TIME_NOW + "/"
    MODEL_DESC:str = FOLDERNAME + "model_" + _fname + "_eps_"+ str(EPOCHS) + "_HOPS_" + str(HOP_STEPS) + "_GVSize_" + str(EMBEDDING_OUTPUT_DIM) + "_DT_" + TIME_NOW + "_"

    #Plotting
    PLOT:str = "plot.png"
    SAVE_PLOTS = True

    _accurracy:list = ['acc']
    _loss_function:str = 'mae'
    _last_activation:str = 'relu'
    _optimizer:str ='adam'
    _use_recursive_encoder:bool = True
    
    _predict_split_value:int = -1
    _dataset_size:int = -1
    _history_keys:list = None
    _max_sequence_len:int = -1
    _unique_words:int = -1
       
    
    # Multi Run Setup!
    _datasets:list = ['2mAMR/2m.json', 'Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt', 'AMR Bio/amr-release-training-bio.txt']
    _multi_epochs:list = [10, 15]
    _multi_hops:list = [6, 9]
    _multi_val_split:list = [0.10, 0.20]
    _runs:int = len(_multi_epochs)

    def Execute(self):
        if self.MULTI_RUN: 
            return self.ExectuteMulti()
        else: 
            return self.ToolPipe()

    def ExectuteMulti(self):
        for dataset in self._datasets:

            self.DATASET_NAME = dataset
            self._fname = dataset.split('/')[0]
            self.DATASET = './Datasets/Raw/'+dataset

            for run in range(self._runs):
                # Reset Logger
                sys.stdout = sys.__stdout__
                sys.stdout.flush()

                # Set network changes
                self.EPOCHS = self._multi_epochs[run]
                self.HOP_STEPS = self._multi_hops[run]
                self.VALIDATION_SPLIT = self._multi_val_split[run]

                #Set name elements and time elements
                self.TIME_NOW:str = strftime("%Y%m%d %H_%M_%S", gmtime())
                self.FOLDERNAME:str = "graph2seq_model_" + self._fname + "_DT_" + self.TIME_NOW + "/"
                self.MODEL_DESC:str = self.FOLDERNAME + "model_" + self._fname + "_eps_"+ str(self.EPOCHS) + "_HOPS_" + str(self.HOP_STEPS) + "_GVSize_" + str(self.EMBEDDING_OUTPUT_DIM) + "_DT_" + self.TIME_NOW + "_"
                self.ToolPipe()

    def SystemInfo(self):
        try:
            print("Model Folder Path:", self.FOLDERNAME)
            if not FolderCreator(self.FOLDERNAME).Create(): 
                print("Result folder was not being created!")
                return False

            sys.stdout = FACLogger(self.FOLDERNAME, self._fname + "_Log")

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
            print("CPUs:\t\t\t=> ", self.CPUS)
            print("GPUs:\t\t\t=> ", self.GPUS)
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
                                    cpu_cores=self.CPUS,
                                    saving_cleaned_data=False,
                                    stringified_amr=semantic_amr_string)

            datapairs = pipe.ProvideData()

            # This part will only be executed if the dataset is provided as matrices.
            if (not semantic_amr_string):
                max_cardinality = pipe._max_observed_nodes_cardinality
                self._dataset_size = len(datapairs)

                pipe.PlotCardinalities(self.MODEL_DESC)
                if self.SHOW_GLOBAL_FEEDBACK:
                    print('Found Datapairs:\n\t=> [', self._dataset_size, '] for allowed graph node cardinality interval [',self.MIN_NODE_CARDINALITY,'|',self.MAX_NODE_CARDINALITY,']')

                MatrixHandler().DatasetLookUpEqualization(datapairs, max_cardinality)
                self._max_sequence_len = (max_cardinality * 2) if max_cardinality != pipe._max_words_sentences else -1

                if self.SHOW_GLOBAL_FEEDBACK:
                    print("~~~~~~ Example Target Data Pipe ~~~~~~~")
                    print("Target 0", datapairs[0][0])
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

                return [max_cardinality, datapairs]
            else:
                print("Prozess of cleaned stringified AMR has ended!")
                sys.exit(0)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.DatasetPreprocessor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def TokenizerPreprocessor(self, datapairs:list, in_vocab_size:int, max_sequence_len:int=-1, show_processor_feedback:bool=True):
        try:
            print("#######################################\n")
            print("########## Dataset Tokenizer ##########")

            glove_dataset_processor = GloVeDatasetPreprocessor( nodes_context=datapairs, 
                                                                vocab_size=in_vocab_size,
                                                                max_sequence_length=max_sequence_len,
                                                                show_feedback=show_processor_feedback)
            _, _, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, nodes_embedding, _ = glove_dataset_processor.Execute()
            self._unique_words = len(glove_dataset_processor._word_index)+1

            if self.SHOW_GLOBAL_FEEDBACK:
                print("Reminder: [1 ----> <go>] and [2 ----> <eos>]")
                print("~~~~~~ Example Target Tokenizer ~~~~~~~")
                glove_dataset_processor.Convert("Input 0", glove_dataset_processor._tokenizer, vectorized_inputs[0])
                glove_dataset_processor.Convert("Target 0", glove_dataset_processor._tokenizer, vectorized_targets[0])
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            glove_dataset_processor.PlotNoneRatio(self.MODEL_DESC)

            return [glove_dataset_processor, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, nodes_embedding]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.TokenizerPreprocessor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)            

    def GloveEmbedding( self, 
                        tokenizer:GloVeDatasetPreprocessor, 
                        nodes_embedding:list, 
                        vectorized_inputs:list, 
                        vectorized_targets:list, 
                        max_cardinality:int, 
                        max_sequence_len:int, 
                        in_vocab_size:int, 
                        out_dim_emb:int, 
                        show_processor_feedback:bool=True, 
                        embedding_input_wordwise:bool=True, 
                        nodes_to_embedding:bool=False):
        try:
            print("#######################################\n")
            print("######## Glove Embedding Layer ########")
            glove_embedding = GloVeEmbedding(   max_cardinality=max_cardinality, 
                                                vocab_size=in_vocab_size, 
                                                tokenizer=tokenizer,
                                                max_sequence_length= max_sequence_len,
                                                glove_file_path=self.GLOVE, 
                                                output_dim=out_dim_emb, 
                                                batch_size=self.BATCH_SIZE,
                                                show_feedback=show_processor_feedback)

            nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(nodes_embedding)
            embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer(embedding_input_wordwise)
            print("Emb_Layer: ", embedding_layer.shape)

            if nodes_to_embedding:
                vectorized_inputs = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_inputs, check_cardinality=False)
                vectorized_targets = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(vectorized_targets, check_cardinality=False)

            if self.SHOW_GLOBAL_FEEDBACK: 
                print("Glove: Result structure [{}{}{}{}]".format(type(nodes_embedding), type(vectorized_inputs), type(vectorized_targets), type(embedding_layer)))

            return [nodes_embedding, vectorized_inputs, vectorized_targets, embedding_layer]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.GloveEmbedding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def CategoricalEmbedding(   self,
                                tokenizer:GloVeDatasetPreprocessor,
                                nodes_embedding:list, 
                                vectorized_inputs:list, 
                                vectorized_targets:list, 
                                max_cardinality:int, 
                                max_sequence_len:int, 
                                in_vocab_size:int,
                                out_dim_emb:int, 
                                show_processor_feedback:bool=True, 
                                embedding_input_wordwise:bool=True,
                                nodes_to_embedding:bool=False):
        try:
            print("#######################################\n")
            print("##### Categorical Embedding Layer #####")
            final_embeddings:list = []
            self.EMBEDDING_OUTPUT_DIM = self._unique_words
            input_len:int = 1 if embedding_input_wordwise else max_sequence_len

            embedding_layer = Embedding(input_dim=self._unique_words, 
                                        output_dim=out_dim_emb, 
                                        input_length=input_len,
                                        trainable=False,
                                        name=('categorical_'+str(out_dim_emb)+'d_embedding'))

            

            if self.SHOW_GLOBAL_FEEDBACK: print("Nodes example: ", nodes_embedding[0])

            nodes_embedding = tokenizer.TokenizeNodes(nodes_embedding)

            # Append 0-vectors to incorrect cardinalities feature arrays

            while nodes_embedding:
                current_features:list = to_categorical(y=nodes_embedding.pop(), num_classes=self._unique_words).tolist()
                diff = max_cardinality - len(current_features)
                for _ in range(diff): current_features.append(np.zeros(self._unique_words))
                final_embeddings.append(np.asarray(current_features))

            if self.SHOW_GLOBAL_FEEDBACK: 
                print("Categorical: Result structure [{}{}{}{}]".format(type(nodes_embedding), type(vectorized_inputs), type(vectorized_targets), type(embedding_layer)))

            return [final_embeddings, vectorized_inputs, vectorized_targets, embedding_layer]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.CategoricalEmbedding]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)

    def DatasetConvertToTeacherForcing(self, nodes_embedding:list, fw_look_up:list, bw_look_up:list, vectorized_inputs:list, vectorized_targets:list, save_dataset:bool=False):
        try:
            print("#######################################\n")
            print("### Create Word Wise Teacherforcing ###")

            
            vectorized_inputs = np.expand_dims(vectorized_inputs, axis=-1)
            vectorized_targets = np.expand_dims(vectorized_targets, axis=-1)

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

    def NetworkInput(self, generator:NumpyDatasetPreprocessor, nodes_embedding:list, fw_look_up, bw_look_up:list, vectorized_inputs:list, vectorized_targets:list):
        try:
            print("#######################################\n")
            print("#### Prepare Train and Predictset #####")
            if isNone(generator):
                generator = NumpyDatasetPreprocessor(None)

            self._dataset_size = len(nodes_embedding)
            self._predict_split_value = DatasetSplitIndex(self._dataset_size, self.PREDICT_SPLIT)

            
            if isList(nodes_embedding): nodes_embedding = np.asarray(nodes_embedding)
            if isList(fw_look_up): fw_look_up = np.asarray(fw_look_up)
            if isList(bw_look_up): bw_look_up = np.asarray(bw_look_up)
            if isList(vectorized_inputs): vectorized_inputs = np.asarray(vectorized_inputs)
            if isList(vectorized_targets): vectorized_targets = np.asarray(vectorized_targets)
            

            # Inputs are numpy.ndarrays and a splitting integer
            train_x, train_y, test_x, test_y = generator.NetworkInputPreparation(   nodes_embedding, 
                                                                                    fw_look_up, 
                                                                                    bw_look_up, 
                                                                                    vectorized_inputs,
                                                                                    vectorized_targets,
                                                                                    self.BATCH_SIZE,
                                                                                    (self._dataset_size - self._predict_split_value))

            # Free space
            nodes_embedding = None
            fw_look_up = None
            bw_look_up = None
            vectorized_inputs = None
            vectorized_targets = None

            if self.SHOW_GLOBAL_FEEDBACK:
                if self.WORD_WISE:
                    print("Train X: ", train_x[0][0].shape, train_x[1][0].shape, train_x[2][0].shape, train_x[3][0].shape)
                    print("Test X: ", test_x[0][0].shape, test_x[1][0].shape, test_x[2][0].shape, test_x[3][0].shape)
                    print("Train Y: ", train_y[0].shape)
                    print("Test Y: ", test_y[0].shape)
                else:
                    print("Train X: ", train_x[0].shape, train_x[1].shape, train_x[2].shape, train_x[3].shape)
                    print("Test X: ", test_x[0].shape, test_x[1].shape, test_x[2].shape, test_x[3].shape)
                    print("Train Y: (" +  str(len(train_y)) + ", " + str(len(train_y[0])) + ")")
                    print("Test Y: (" + str(len(test_y)) + ", " + str(len(test_y[0])) + ")")
                print("Network Input: Result structure [{}{}{}{}]".format(type(train_x), type(train_y), type(test_x), type(test_y)))

            return [train_x, train_y, test_x, test_y]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkInput]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex) 

    def NetworkGraphEncoderConstruction(self, target_shape:tuple, max_cardinality:int, embedding_layer:keras.layers.embeddings.Embedding):
        try:
            input_dec_dim = 1 if self.WORD_WISE else target_shape[-1]

            print("Builder Init!")
            builder = ModelBuilder( input_enc_dim=self.EMBEDDING_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=input_dec_dim,
                                    batch_size=self.BATCH_SIZE)

            print("Build Neighbourhood Submodel!")
            graph_embedding, graph_embedding_h, graph_embedding_c = builder.BuildGraphEmbeddingLayers(hops=self.HOP_STEPS, hidden_dim=self.EMBEDDING_OUTPUT_DIM)
            sequence_embedding = embedding_layer(builder.get_decoder_inputs())

            if ((not self._use_recursive_encoder) and (self.WORD_WISE)):
                print("Build Encoder Stated!")
                units, encoder, enc_h, enc_c, att_weights = builder.BuildStatePassingEncoder(   sequence_embedding=sequence_embedding,
                                                                                                graph_embedding=graph_embedding,
                                                                                                prev_memory_state=graph_embedding_h,  
                                                                                                prev_carry_state=graph_embedding_c)
                return [builder, units, encoder, enc_h, enc_c, att_weights]
            else:
                print("Build Encoder Repeated!")
                units, encoder = builder.BuildRecursiveEncoder( sequence_lenght=(1 if self.WORD_WISE else self._max_sequence_len),
                                                                sequence_embedding=sequence_embedding,
                                                                graph_embedding= graph_embedding,
                                                                prev_memory_state=graph_embedding_h,  
                                                                prev_carry_state=graph_embedding_c)
                return [builder, units, encoder, None, None, None]
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkGraphEncoderConstruction]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def NetworkDecoderConstruction(self, builder:ModelBuilder, encoder, units:int, encoder_states):
        try:
            print("Build Decoder!")
            decoder = builder.BuildDecoder( units=units, 
                                            encoder=encoder,
                                            prev_memory_state=encoder_states[0],  
                                            prev_carry_state=encoder_states[1])
            return builder.BuildDecoderPrediction(previous_layer=decoder, act=self._last_activation)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkDecoderConstruction]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def NetworkConstruction(self, target_shape:tuple, max_cardinality:int, embedding_layer:keras.layers.embeddings.Embedding):
        try:
            print("#######################################\n")
            print("####### Construct Network Model #######")
            builder, units, encoder, enc_h, enc_c, _ = self.NetworkGraphEncoderConstruction(target_shape, max_cardinality, embedding_layer)
            model = self.NetworkDecoderConstruction(builder, encoder, units, [enc_h, enc_c])


            print("Build Finalize and Plot!")
            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model, optimizer=self._optimizer, metrics=self._accurracy, loss = self._loss_function)
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
            #mc = ModelCheckpoint(self.MODEL_DESC+'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


            return model.fit(   train_x, 
                                train_y,
                                batch_size = self.BATCH_SIZE,
                                epochs=self.EPOCHS, 
                                verbose=self.VERBOSE, 
                                shuffle=self.SHUFFLE_DATASET,
                                validation_split=self.VALIDATION_SPLIT,
                                callbacks=[base_lr, reduce_lr, es])
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkTrain]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  

    def NetworkPlotResults(self, history, iterator_value:int = -1, new_style:bool = False):
        try:
            print("#######################################\n")
            print("######## Plot Training Results ########")

            print(type(history))
            if self.SHOW_GLOBAL_FEEDBACK:
                print("History Keys: ", list(history.history.keys()))

            description:str = self.MODEL_DESC if (iterator_value < 0) else self.MODEL_DESC + "_Num_" + str(iterator_value)
            plotter = HistoryPlotter(   model_description = description, 
                                        path = None, 
                                        history = history,
                                        new_style = new_style)

            plotter.PlotHistory()
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.NetworkPlotResults]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)  
    
    def NetworkPredict(self, model, test_x:list, test_y:np.ndarray):
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

            max_cardinality, datapairs = self.DatasetPreprocessor(  in_dataset=self.DATASET,
                                                                    in_extender=self.EXTENDER,
                                                                    in_max_length=self.MAX_LENGTH_DATA, 
                                                                    keep_edges=self.KEEP_EDGES)
            
            tokenizer, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets, nodes_embedding = self.TokenizerPreprocessor( datapairs=datapairs, 
                                                                                                                                    in_vocab_size=self.GLOVE_VOCAB_SIZE, 
                                                                                                                                    max_sequence_len=self._max_sequence_len,
                                                                                                                                    show_processor_feedback=True)
            embedding_layer = None

            if self.USE_GLOVE:
                nodes_embedding, vectorized_inputs, vectorized_targets, embedding_layer = self.GloveEmbedding(  tokenizer=tokenizer, 
                                                                                                                nodes_embedding=nodes_embedding, 
                                                                                                                vectorized_inputs=vectorized_inputs, 
                                                                                                                vectorized_targets=vectorized_targets, 
                                                                                                                max_cardinality=max_cardinality, 
                                                                                                                max_sequence_len=self._max_sequence_len, 
                                                                                                                in_vocab_size=self.GLOVE_VOCAB_SIZE, 
                                                                                                                out_dim_emb=self.EMBEDDING_OUTPUT_DIM, 
                                                                                                                show_processor_feedback=True, 
                                                                                                                embedding_input_wordwise=self.WORD_WISE, 
                                                                                                                nodes_to_embedding=False)
            else:
                nodes_embedding, vectorized_inputs, vectorized_targets, embedding_layer = self.CategoricalEmbedding(tokenizer=tokenizer,
                                                                                                                    nodes_embedding=nodes_embedding,
                                                                                                                    vectorized_inputs=vectorized_inputs, 
                                                                                                                    vectorized_targets=vectorized_targets, 
                                                                                                                    max_cardinality=max_cardinality, 
                                                                                                                    max_sequence_len=self._max_sequence_len, 
                                                                                                                    in_vocab_size=self.GLOVE_VOCAB_SIZE,
                                                                                                                    out_dim_emb=self.EMBEDDING_OUTPUT_DIM,
                                                                                                                    show_processor_feedback=True, 
                                                                                                                    embedding_input_wordwise=self.WORD_WISE, 
                                                                                                                    nodes_to_embedding=False)
            tokenizer = None
            generator = None
            
            if self.WORD_WISE:
                generator, nodes_embedding, fw_look_up, bw_look_up, vectorized_inputs, vectorized_targets = self.DatasetConvertToTeacherForcing(nodes_embedding=nodes_embedding, 
                                                                                                                                                fw_look_up=fw_look_up, 
                                                                                                                                                bw_look_up=bw_look_up, 
                                                                                                                                                vectorized_inputs=vectorized_inputs, 
                                                                                                                                                vectorized_targets=vectorized_targets, 
                                                                                                                                                save_dataset=False)

            train_x, train_y, test_x, test_y = self.NetworkInput(   generator=generator, 
                                                                    nodes_embedding=nodes_embedding, 
                                                                    fw_look_up=fw_look_up, 
                                                                    bw_look_up=bw_look_up, 
                                                                    vectorized_inputs=vectorized_inputs, 
                                                                    vectorized_targets=vectorized_targets)

            model = self.NetworkConstruction(   target_shape=vectorized_targets.shape, 
                                                max_cardinality=max_cardinality, 
                                                embedding_layer=embedding_layer)

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

    
if __name__ == "__main__":
    Graph2SeqInKeras().Execute()