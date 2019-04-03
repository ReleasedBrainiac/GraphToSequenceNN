import argparse
import os, sys
import platform as pf
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import History
import matplotlib.pyplot as plt

from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import RoundUpRestricted
from GloVeHandler.GloVeDatasetPreprocessor import GloVeDatasetPreprocessor
from GloVeHandler.GloVeEmbedding import GloVeEmbedding
from DatasetHandler.FileWriter import Writer
from DatasetHandler.ContentSupport import MatrixExpansionWithZeros
from NetworkHandler.Builder.ModelBuilder import ModelBuilder

#TODO IN MA => Ausblick => https://github.com/philipperemy/keras-attention-mechanism
#TODO IN MA => Ausblick => https://github.com/keras-team/keras/issues/4962
#TODO IN MA => Code => Expansion of edge matrices why? => Layers weights!
#TODO IN MA => Code => Why min and max cardinality
#TODO IN MA => Resource for MA and Code ~> https://stackoverflow.com/questions/32771786/predictions-using-a-keras-recurrent-neural-network-accuracy-is-always-1-0/32788454#32788454 
#TODO IN MA => Best LSTM resources ~> https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
#TODO IN MA => 2nd best LSTM resource ~> https://adventuresinmachinelearning.com/keras-lstm-tutorial/

#TODO Guide to finalize Tool => https://keras.io/examples/lstm_stateful/
#TODO Guide to finalize Tool => https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/
#TODO Guide to finalize Tool => https://keras.io/models/model/#fit 
#TODO Guide to finalize Tool => https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
#TODO Guide to finalize Tool => https://keras.io/visualization/#training-history-visualization

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
    EPOCHS:int = 5
    VERBOSE:int = 1
    BATCH_SIZE:int = 1
    BUILDTYPE:int = 1
    DATASET_NAME:str = 'AMR Bio/amr-release-training-bio.txt'    #'Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
    fname = DATASET_NAME.split('/')[0]
    DATASET:str = './Datasets/Raw/'+DATASET_NAME
    GLOVE:str = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    GLOVE_VEC_SIZE:int = 100
    MODEL:str = "graph2seq_model"
    PLOT:str = "plot.png"
    EXTENDER:str = "dc.ouput"
    MAX_LENGTH_DATA:int = -1
    SHOW_FEEDBACK:bool = False
    STORE_STDOUT:bool = False
    SAVE_PLOTS = False
    SAVING_CLEANED_AMR:bool = False
    KEEP_EDGES:bool = True
    GLOVE_OUTPUT_DIM:int = 100
    GLOVE_VOCAB_SIZE:int = 5000
    VALIDATION_SPLIT:float = 0.2
    MIN_NODE_CARDINALITY:int = 3
    MAX_NODE_CARDINALITY:int = 19
    HOP_STEPS:int = 3
    BIDIRECTIONAL_MULT:int = 2
    SHUFFLE_DATASET:bool = True

    def ExecuteTool(self):
        """
        The main method of the tool.
        It provides 2 functions:
            1. Storing the cleaned version of the passed AMR file
            2. Execute the network on the given dataset (includes cleaning but no storing of the AMR). 
        """  
        try:
            if (self.STORE_STDOUT): sys.stdout = open(self.fname+' console_report.txt', 'w')

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
                                max_cardinality=self.MAX_NODE_CARDINALITY
                                )
            datapairs = pipe.ProvideData()
            max_cardinality = pipe.max_observed_nodes_cardinality

            print('Found Datapairs:\n\t=> [', len(datapairs), '] for allowed graph node cardinality interval [',self.MIN_NODE_CARDINALITY,'|',self.MAX_NODE_CARDINALITY,']')
            pipe.ShowNodeCardinalityOccurences()
            self.DatasetLookUpEqualization(datapairs, max_cardinality)
                

            print("#######################################\n")
            print("######## Glove Embedding Layer ########")
            glove_dataset_processor = GloVeDatasetPreprocessor( nodes_context=datapairs, 
                                                                vocab_size=in_vocab_size,
                                                                max_sequence_length=pipe.max_sentences,
                                                                show_feedback=True)
            _, _, edge_fw_look_up, edge_bw_look_up, vectorized_sequences, vectorized_targets, dataset_nodes_values, dataset_indices = glove_dataset_processor.Execute()
            
            glove_embedding = GloVeEmbedding(   max_cardinality=max_cardinality, 
                                                vocab_size=in_vocab_size, 
                                                tokenizer=glove_dataset_processor,
                                                max_sequence_length=pipe.max_sentences,
                                                glove_file_path=self.GLOVE, 
                                                output_dim=out_dim_emb, 
                                                show_feedback=True)
            datasets_nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(dataset_nodes_values)
            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer()

            print('Embedding Resources:\n\t => Free (in further steps unused) resources!', )
            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()
                
            print("#######################################\n")
            print("########## Construct Network ##########")

            builder = ModelBuilder( input_enc_dim=self.GLOVE_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=pipe.max_sentences,
                                    batch_size=self.BATCH_SIZE,
                                    input_is_2d=False)

            _, graph_embedding_encoder_states = builder.BuildGraphEmbeddingEncoder(hops=self.HOP_STEPS)

            model = builder.BuildGraphEmbeddingDecoder( embedding_layer=glove_embedding_layer(builder.get_decoder_inputs()),
                                                        prev_memory_state=graph_embedding_encoder_states[0],  
                                                        prev_carry_state=graph_embedding_encoder_states[1])

            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model)
            if self.SHOW_FEEDBACK: builder.Summary(model)
            builder.Plot(model=model, file_name=self.fname+'_model_graph.png')

            print("#######################################\n")
            print("########### Starts Training ###########")

            history = model.fit([datasets_nodes_embedding, edge_fw_look_up, edge_bw_look_up, vectorized_sequences], 
                                vectorized_targets,
                                batch_size = self.BATCH_SIZE,
                                epochs=self.EPOCHS, 
                                verbose=self.VERBOSE, 
                                shuffle=self.SHUFFLE_DATASET,
                                validation_split=self.VALIDATION_SPLIT)

            print("#######################################\n")
            print("############# Save Model ##############")

            model.save_weights(self.fname+'_graph_2_sequence_trained_weights.h5')

            print("#######################################\n")
            print("######## Plot Training Results ########")

            plt.plot(history.history['top_k_categorical_accuracy'])
            plt.plot(history.history['val_top_k_categorical_accuracy'])
            plt.title('Model Top k Categorical Accuracy')
            plt.ylabel('Top k Categorical Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            if not self.SAVE_PLOTS: 
                self.SavePyPlotToFile(extender='top_k_categoriacal_epoch_plot')
            else: 
               plt.show()

            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
            plt.title('Model Categorical Accuracy')
            plt.ylabel('Categorical Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            if not self.SAVE_PLOTS: 
                self.SavePyPlotToFile(extender='categoriacal_epoch_plot')
            else: 
               plt.show()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            if not self.SAVE_PLOTS: 
                self.SavePyPlotToFile(extender='loss_epoch_plot')
            else: 
                plt.show()

            #TODO Maybe add Predict
            #print("#######################################\n")
            #print("######## Predict Model ########")

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
                plt.savefig((self.fname+'_plot.'+image_type), orientation=orientation)
            else: 
                plt.savefig((self.fname+'_'+extender+'.'+image_type), orientation=orientation)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.SavePyPlotToFile]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(ex)
    
if __name__ == "__main__":
    Graph2SeqInKeras().ExecuteTool()