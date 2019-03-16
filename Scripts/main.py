import argparse
import os, sys
import platform as pf
import numpy as np
import tensorflow as tf
import keras
from subprocess import call

from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import ReorderListByIndices
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




'''
#TODO Packages missing reviews!
#TODO NetworkHandler
#TODO GloVeHandler

'''

class Graph2SequenceTool():

    TF_CPP_MIN_LOG_LEVEL="2"
    EPOCHS = 50
    BATCH_SIZE = 15
    BUILDTYPE = 1
    DATASET = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
    GLOVE = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    GLOVE_VEC_SIZE = 100
    MODEL = "graph2seq_model"
    PLOT = "plot.png"
    EXTENDER = "dc.ouput"
    MAX_LENGTH_DATA = -1
    SHOW_FEEDBACK = False
    SAVING_CLEANED_AMR = False
    KEEP_EDGES = True
    GLOVE_OUTPUT_DIM = 100
    GLOVE_VOCAB_SIZE = 20000
    VALIDATION_SPLIT = 0.2
    MIN_NODE_CARDINALITY = 3
    MAX_NODE_CARDINALITY = 19
    HOP_STEPS = 3

    def RunTool(self):
        """
        This is the main method of the tool.
        """  
        try:
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
                self.RunStoringCleanedAMR(in_dataset=self.DATASET,
                                          in_extender=self.EXTENDER,
                                          in_max_length=self.MAX_LENGTH_DATA,
                                          is_show=self.SHOW_FEEDBACK,
                                          keep_edges=self.KEEP_EDGES,
                                          as_amr=True)
            else:
                self.RunTrainProcess(in_dataset=self.DATASET, 
                                     in_glove=self.GLOVE, 
                                     in_extender=self.EXTENDER,
                                     in_max_length=self.MAX_LENGTH_DATA,
                                     in_vocab_size=self.GLOVE_VOCAB_SIZE,
                                     out_dim_emb=self.GLOVE_OUTPUT_DIM,
                                     is_show=self.SHOW_FEEDBACK,
                                     keep_edges=self.KEEP_EDGES)

        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.RunTool]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    #TODO maybe rename cause train, test and predict will be solved at once
    
    def RunTrainProcess(self, in_dataset, in_glove, in_extender="output", in_max_length=-1, in_vocab_size=20000, out_dim_emb=100, is_show=True, keep_edges=False):
        """
        This function execute the training pipeline for the graph2seq tool.
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
            #TODO check switches!
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

            print('Embeddings:', datasets_nodes_embedding.shape)

            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer()

            print('Embedding Resources:\n\t => Free (in further steps unused) resources!', )
            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()

            #TODO insert missing dimension for Encoder Model

            print('DS_Nodes_Emb:\n', datasets_nodes_embedding.shape)
            print('Inp_Vec:\n', vectorized_sequences.shape, ' | ', vectorized_sequences[0].shape )
            print('Tar_Vec:\n', vectorized_targets.shape, ' | ', vectorized_targets[0].shape)
                


            print("#######################################\n")
            print("######### Random Order Dataset ########")

            if (self.SHOW_FEEDBACK): 
                print('Old order \t => ', dataset_indices)

            np.random.shuffle(dataset_indices)

            if (self.SHOW_FEEDBACK): 
                print('New order \t => ', dataset_indices)

            network_input_sentences = vectorized_sequences[dataset_indices]
            network_input_targets = vectorized_targets[dataset_indices]
            network_input_graph_features = datasets_nodes_embedding[dataset_indices]
            network_input_fw_look_up = edge_fw_look_up[dataset_indices]
            network_input_bw_look_up = edge_bw_look_up[dataset_indices]


            print("#######################################\n")
            print("############ Split Dataset ############")
            samples_size = len(network_input_fw_look_up)
            print('Samples \t => ', samples_size)

            nb_validation_samples = int(self.VALIDATION_SPLIT * samples_size)
            print('Validation \t => ', nb_validation_samples)

            x_train_edge_fw = network_input_fw_look_up[:-nb_validation_samples]
            x_train_edge_bw = network_input_bw_look_up[:-nb_validation_samples]
            x_train_features = network_input_graph_features[:-nb_validation_samples]
            y_train_sentences = network_input_sentences[:-nb_validation_samples]
            y_train_targets = network_input_targets[:-nb_validation_samples]
            

            if(self.SHOW_FEEDBACK): 
                print('x_train_edge_fw: ', type(x_train_edge_fw), x_train_edge_fw.shape)
                print('x_train_edge_bw: ', type(x_train_edge_bw), x_train_edge_bw.shape)
                print('x_train_features: ', type(x_train_features), x_train_features.shape)
                print('y_train_sentences: ', type(y_train_sentences), y_train_sentences.shape)
                print('y_train_targets: ', type(y_train_targets), y_train_targets.shape)

            print('Train set \t =>  defined!')

            x_validation_edge_fw = network_input_fw_look_up[-nb_validation_samples:]
            x_validation_edge_bw = network_input_bw_look_up[-nb_validation_samples:]
            x_validation_features = network_input_graph_features[-nb_validation_samples:]
            y_validation_sentences = network_input_sentences[-nb_validation_samples:]
            y_validation_targets = network_input_targets[-nb_validation_samples:]

            if(self.SHOW_FEEDBACK): 
                print('x_validation_edge_fw: ', type(x_validation_edge_fw), x_validation_edge_fw.shape)
                print('x_validation_edge_bw: ', type(x_validation_edge_bw), x_validation_edge_bw.shape)
                print('x_validation_features: ', type(x_validation_features), x_validation_features.shape)
                print('y_validation_sentences: ', type(y_validation_sentences), y_validation_sentences.shape)
                print('y_validation_targets: ', type(y_validation_targets), y_validation_targets.shape)

            print('Test set \t =>  defined!')

            print("#######################################\n")
            print("########## Construct Network ##########")

            builder = ModelBuilder( input_enc_dim=self.GLOVE_OUTPUT_DIM, 
                                    edge_dim=max_cardinality, 
                                    input_dec_dim=pipe.max_sentences)

            _, graph_embedding_encoder_states = builder.GraphEmbeddingEncoderBuild(hops=self.HOP_STEPS)

            model = builder.GraphEmbeddingDecoderBuild( embedding_layer=glove_embedding_layer(builder.get_decoder_inputs()),
                                                        prev_memory_state=graph_embedding_encoder_states[0],  
                                                        prev_carry_state=graph_embedding_encoder_states[1])

            model = builder.MakeModel(layers=[model])
            builder.CompileModel(model=model)
            builder.Summary(model)
            builder.Plot(model=model, file_name='model_graph.png')


            print("#######################################\n")
            print("########### Starts Training ###########")

            
            ''' model.fit([], 
                      [], 
                      steps_per_epoch=1, 
                      validation_steps=1, 
                      epochs=1, 
                      verbose=0, 
                      shuffle=False)
            '''
            

            print("#######################################\n")
            print("############# Save Model ##############")

            print("#######################################\n")
            print("######## Plot Training Results ########")

        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.RunTrainProcess]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def RunStoringCleanedAMR(self, 
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
            template = "An exception of type {0} occurred in [Main.RunStoringCleanedAMR]. Arguments:\n{1!r}"
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

if __name__ == "__main__":
    Graph2SequenceTool().RunTool()