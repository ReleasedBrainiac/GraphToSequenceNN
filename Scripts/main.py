import argparse
import os, sys
import platform as pf
import numpy as np
from subprocess import call

from DatasetHandler.DatasetProvider import DatasetPipeline
from DatasetHandler.ContentSupport import ReorderListByIndices
from GloVeHandler.GloVeDatasetPreprocessor import GloVeDatasetPreprocessor
from GloVeHandler.GloVeEmbedding import GloVeEmbedding
from NetworkHandler.LayerBuilder import CustomLayerDefinitions
from DatasetHandler.FileWriter import Writer

class Graph2SequenceTool():

    TF_CPP_MIN_LOG_LEVEL="2"
    EPOCHS = 50
    BATCH_SIZE = 15
    BUILDTYPE = 1
    DATASET = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
    GLOVE = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    MODEL = "graph2seq_model"
    PLOT = "plot.png"
    EXTENDER = "dc.ouput"
    MAX_LENGTH_DATA = -1
    SHOW_FEEDBACK = False
    SAVING_CLEANED_AMR = False
    KEEP_EDGES = False
    GLOVE_OUTPUT_DIM = 100
    GLOVE_VOCAB_SIZE = 20000
    VALIDATION_SPLIT = 0.2

    def RunTool(self):
        """
        This is the main method of the tool.
        """  
        try:
            print("\n#######################################")
            print("######## Graph to Sequence ANN ########")
            print("#######################################\n")

            print("System we running on is ", pf.system())
            print("Release: ", pf.release())

            print("Parsing command-line attributes!")
            ap = argparse.ArgumentParser()
            ap.add_argument("-dp"   , "--dataset_path"      , type=str      , required=False, default=self.DATASET                , help="Path for the root folder of the dataset!")
            ap.add_argument("-gp"   , "--glove_path"        , type=str      , required=False, default=self.GLOVE                  , help="Path for the root folder glove word vector file!")
            ap.add_argument("-mn"   , "--model_name"        , type=str      , required=False, default=self.MODEL                  , help="Name of your model. Named paths are also allowed!")
            ap.add_argument("-pltp" , "--plot_path"         , type=str      , required=False, default=self.PLOT                   , help="Name of the plotted results [name].png. Named paths are also allowed!")
            ap.add_argument("-eps"  , "--epochs"            , type=int      , required=False, default=self.EPOCHS                 , help="Amount of epochs for the network training process!")
            ap.add_argument("-bats" , "--batch_size"        , type=int      , required=False, default=self.BATCH_SIZE             , help="Size of batches for the network training process!")
            ap.add_argument("-god"  , "--glove_out_dim"     , type=int      , required=False, default=self.GLOVE_OUTPUT_DIM       , help="Dimension of the glove embedding output! This has to be equel to the used Glove file definition!")
            ap.add_argument("-gvs"  , "--glove_vocab_size"  , type=int      , required=False, default=self.GLOVE_VOCAB_SIZE       , help="Amount of most words the embedding should keep!The rest will be mapped to zero!")
            ap.add_argument("-dvs"  , "--dataset_val_split" , type=float    , required=False, default=self.VALIDATION_SPLIT       , help="Spitting percentage of the dataset for treaining and testing!")
            #ap.add_argument("-btype", "--build_type"        , type=int      , required=False, default=self.BUILDTYPE              , help="Build type for the network training process!")
            ap.add_argument("-tll"  , "--tf_min_log_lvl"    , type=str      , required=False, default=self.TF_CPP_MIN_LOG_LEVEL   , help="Tensorflow logging level!")
            ap.add_argument("-ext"  , "--output_extender"   , type=str      , required=False, default=self.EXTENDER               , help="File extension element for cleaned amr output files!")
            ap.add_argument("-mlen" , "--max_length"        , type=int      , required=False, default=self.MAX_LENGTH_DATA        , help="Max size for amr dataset sentences (the doubled value defines the graph maximum lenght then)!")
            ap.add_argument("-sfb"  , "--show_feedback"     , type=bool     , required=False, default=self.SHOW_FEEDBACK          , help="Show processing feedback on the console or not!")
            ap.add_argument("-scd"  , "--save_cleaned_data" , type=bool     , required=False, default=self.SAVING_CLEANED_AMR     , help="Save the cleaned amr data and end the process or process the full pipe!")
            ap.add_argument("-ke"   , "--keep_edges"        , type=bool     , required=False, default=self.KEEP_EDGES             , help="Allow to keep the edges in the cleaning process! Note that the further edge processing is not implemtend for tensorflow usage!")


            args = vars(ap.parse_args())

            self.DATASET = args["dataset_path"]
            self.GLOVE = args["glove_path"]
            self.MODEL = args["model_name"]
            self.PLOT = args["plot_path"]
            self.EPOCHS = args["epochs"]
            self.BATCH_SIZE = args["batch_size"]
            self.GLOVE_OUTPUT_DIM = args["glove_out_dim"]
            self.GLOVE_VOCAB_SIZE = args["glove_vocab_size"]
            self.VALIDATION_SPLIT = args["dataset_val_split"]
            #self.BUILDTYPE = args["build_type"]
            self.TF_CPP_MIN_LOG_LEVEL = args["tf_min_log_lvl"]
            self.EXTENDER = args["output_extender"]
            self.MAX_LENGTH_DATA = args["max_length"]
            self.SHOW_FEEDBACK = args["show_feedback"]
            self.SAVING_CLEANED_AMR = args["save_cleaned_data"]
            self.KEEP_EDGES = args["keep_edges"]
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.TF_CPP_MIN_LOG_LEVEL


            if self.SAVING_CLEANED_AMR:
                self.RunStoringCleanedAMR(in_dataset=self.DATASET,
                                          in_extender=self.EXTENDER,
                                          in_max_length=self.MAX_LENGTH_DATA,
                                          is_show=self.SHOW_FEEDBACK,
                                          is_keeping_edges=self.KEEP_EDGES,
                                          is_amr_saving=True)
            else:
                self.RunTrainProcess(in_dataset=self.DATASET, 
                                     in_glove=self.GLOVE, 
                                     in_extender=self.EXTENDER,
                                     in_max_length=self.MAX_LENGTH_DATA,
                                     in_vocab_size=self.GLOVE_VOCAB_SIZE,
                                     out_dim_emb=self.GLOVE_OUTPUT_DIM,
                                     is_show=self.SHOW_FEEDBACK,
                                     is_keeping_edges=self.KEEP_EDGES)

        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.RunTool]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    #TODO maybe rename cause train, test and predict will be solved at once
    def RunTrainProcess(self, in_dataset, in_glove, in_extender="output", in_max_length=-1, in_vocab_size=20000, out_dim_emb=100, is_show=True, is_keeping_edges=False):
        """
        This function execute the training pipeline for the graph2seq tool.
            :param in_dataset: input dataset
            :param in_glove: path to glove word vector file
            :param in_extender: output file entension
            :param in_max_length: max allowed length of sentences (max_len Semantic = 2x max_len sentences)
            :param in_vocab_size: the max amount of most occouring words you desire to store
            :param out_dim_emb: dimension for the glove embedding layer output dimension
            :param is_show: show process steps as console feedback
            :param is_keeping_edges: keep edgesduring cleaning
        """   
        try:
            print("#######################################\n")
            print("###### AMR Dataset Preprocessing ######")
            
            pipe = DatasetPipeline(in_path=in_dataset, 
                                output_path_extender=in_extender, 
                                max_length=in_max_length, 
                                show_feedback=is_show, 
                                saving=False, 
                                keep_edges=is_keeping_edges
                                )
            datapairs = pipe.ProvideData()

            print('Found Datapairs: ', len(datapairs))

            print("#######################################\n")
            print("######## Glove Embedding Layer ########")
            #TODO check switches!
            glove_dataset_processor = GloVeDatasetPreprocessor(nodes_context=datapairs, vocab_size=in_vocab_size, show_feedback=True)
            datasets_sentences, directed_edge_matrices, vectorized_sequences, dataset_nodes_values, dataset_indices = glove_dataset_processor.GetPreparedDataSamples()

            glove_embedding = GloVeEmbedding(vocab_size=in_vocab_size, tokenizer=glove_dataset_processor, glove_file_path=self.GLOVE, output_dim=out_dim_emb, show_feedback=True)
            datasets_nodes_embedding = glove_embedding.ReplaceDatasetsNodeValuesByEmbedding(dataset_nodes_values)
            glove_embedding_layer = glove_embedding.BuildGloveVocabEmbeddingLayer()

            print('\t => Free (in further steps unused) resources from embedding process!', )
            glove_embedding.ClearTokenizer()
            glove_embedding.ClearEmbeddingIndices()


            print("#######################################\n")
            print("######### Random Order Dataset ########")

            np.random.shuffle(dataset_indices)
            network_input_sentences = vectorized_sequences[dataset_indices]
            network_input_graph_features = datasets_nodes_embedding[dataset_indices]      
            network_input_directed_edge_matrices = ReorderListByIndices(directed_edge_matrices, dataset_indices)


            print("#######################################\n")
            print("############ Split Dataset ############")

            nb_validation_samples = int(self.VALIDATION_SPLIT * len(network_input_directed_edge_matrices))
            print('Samples: ', nb_validation_samples)

            x_train_edge = network_input_directed_edge_matrices[:-nb_validation_samples]
            x_train_features = network_input_graph_features[:-nb_validation_samples]
            y_train_sentences = network_input_sentences[:-nb_validation_samples]
            print('Train set!')


            x_validation_edge = network_input_directed_edge_matrices[-nb_validation_samples:]
            x_validation_features = network_input_graph_features[-nb_validation_samples:]
            y_validation_sentences = network_input_sentences[-nb_validation_samples:]
            print('Test set!')

            print('x_train_edge: ', type(x_train_edge), '\n',x_train_edge[0])
            print('x_train_features: ', type(x_train_features), '\n',x_train_features[0])
            print('y_train_sentences: ', type(y_train_sentences), '\n',y_train_sentences[0])
            print('x_validation_edge: ', type(x_validation_edge), '\n',x_validation_edge[0])
            print('x_validation_features: ', type(x_validation_features), '\n',x_validation_features[0])
            print('y_validation_sentences: ', type(y_validation_sentences), '\n',y_validation_sentences[0])




            print('Shape features: ', x_train_features.shape[0])


            print("#######################################\n")
            print("######## Nodes Embedding Layer ########")
        

            print("#######################################\n")
            print("########## Construct Network ##########")


            print("#######################################\n")
            print("########### Starts Training ###########")

            '''
            model.fit(test_in, 
                      test_in, 
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

    def RunStoringCleanedAMR(self, in_dataset, in_extender, in_max_length, is_show, is_keeping_edges, is_amr_saving):
        """
        This function stores the cleaned amr dataset, only!
            :param in_dataset: input dataset
            :param in_extender: output file entension
            :param in_max_length: max allowed length of sentences (max_len Semantic = 2x max_len sentences)
            :param is_show: show process steps as console feedback
            :param is_keeping_edges: keep edgesduring cleaning
            :param is_amr_saving: save as amr if true else as json
        """   
        try:
            pipe = DatasetPipeline(in_path=in_dataset, 
                            output_path_extender=in_extender, 
                            max_length=in_max_length, 
                            show_feedback=is_show, 
                            saving=True, 
                            keep_edges=is_keeping_edges
                            )

            pipe.SaveData(as_amr=is_amr_saving)
            print('Finished storing process!')
        except Exception as ex:
            template = "An exception of type {0} occurred in [Main.RunStoringCleanedAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

if __name__ == "__main__":
    Graph2SequenceTool().RunTool()