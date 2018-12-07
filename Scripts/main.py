# - *- coding: utf- 8*-

import argparse
import os
import platform as pf
from subprocess import call

from DatasetHandler.DatasetProvider import DatasetPipeline
from GloVeHandler.GloVeDatasetParser import GloVeEmbedding

def main():

    TF_CPP_MIN_LOG_LEVEL="2"
    EPOCHS = 50
    BATCH_SIZE = 15
    BUILDTYPE = 1
    DATASET = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
    GLOVE = './Datasets/GloVeWordVectors/glove.6B/glove.6B.100d.txt'
    MODEL = "graph2seq_model"
    PLOT = "plot.png"

    print("\n#######################################n")
    print(  "######## Graph to Sequence ANN ########\n")
    print(  "#######################################\n")

    if(pf.system() == "Windows"): call(["chcp","65001"])
    print("System we running on is ", pf.system())
    print("Release: ", pf.release())

    print("Parsing command-line attributes!")
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp"   , "--dataset_path"      , type=str      , required=True, default=DATASET                , help="Path for the root folder of the dataset!")
    ap.add_argument("-gp"   , "--glove_path"        , type=str      , required=True, default=GLOVE                  , help="Path for the root folder glove word vector file!")
    ap.add_argument("-mn"   , "--model_name"        , type=str      , required=True, default=MODEL                  , help="Name of your model. Named paths are also allowed!")
    ap.add_argument("-pltp" , "--plot_path"         , type=str      , required=True, default=PLOT                   , help="Name of the plotted results [name].png. Named paths are also allowed!")
    ap.add_argument("-eps"  , "--epochs"            , type=int      , required=True, default=EPOCHS                 , help="Amount of epochs for the network training process!")
    ap.add_argument("-bats" , "--batch_size"        , type=int      , required=True, default=BATCH_SIZE             , help="Size of batches for the network training process!")
    #ap.add_argument("-btype", "--build_type"        , type=int      , required=True, default=BUILDTYPE              , help="Build type for the network training process!")
    ap.add_argument("-tll"  , "--tf_min_log_lvl"    , type=str      , required=True, default=TF_CPP_MIN_LOG_LEVEL   , help="Tensorflow logging level!")
    

    args = vars(ap.parse_args())

    DATASET = args["dataset_path"]
    GLOVE = args["glove_path"]
    MODEL = args["model_name"]
    PLOT = args["plot_path"]
    EPOCHS = args["epochs"]
    BATCH_SIZE = args["batch_size"]
    #BUILDTYPE = args["build_type"]
    TF_CPP_MIN_LOG_LEVEL = args["tf_min_log_lvl"]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

    print("#######################################\n")
    print("### Preprocessing the raw datesets! ###")

    pipe = DatasetPipeline(in_path=DATASET, 
                        output_path_extender='dc.ouput', 
                        max_length=-1, 
                        show_feedback=False, 
                        saving=False, 
                        keep_edges=False
                        )

    datapairs = pipe.ProvideData()
    embedding = GloVeEmbedding(nodes_context=datapairs, vocab_size=400000 ,glove=GLOVE, output_dim=100, show_feedback=True).BuildGloveVocabEmbeddingLayer()

    print("Build feature vectors!")

    print("Construct model!")

    print("Summary of the constructed model!")

    print("Train model!")

    print("Save model, weights and plot!")

    print("Predict with model!")



if __name__ == "__main__":
    main()
