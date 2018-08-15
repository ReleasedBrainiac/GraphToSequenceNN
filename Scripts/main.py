# - *- coding: utf- 8*-

import argparse
import os
import platform as pf
from subprocess import call

def main():

    TF_CPP_MIN_LOG_LEVEL="2"
    EPOCHS = 50
    BATCH_SIZE = 15
    BUILDTYPE = 1

    print("\n#########################################\n")
    print(  "########### Context Extractor ###########\n")
    print(  "#########################################\n")

    if(pf.system() == "Windows"):
        call(["chcp","65001"])

    print("System we running on is ", pf.system())
    print("Release: ", pf.release())

    print("Parsing command-line attributes!")
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="Path for the root folder of the raw training dataset!")
    ap.add_argument("-v", "--validation", required=True, help="Path for the root folder of the raw validation dataset!")
    ap.add_argument("-m", "--model_name", required=True, help="Name of your model. Named paths are also allowed!")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Name of the plotted results [name].png. Named paths are also allowed!")
    args = vars(ap.parse_args())

    trainPath = args["training"]
    validPath = args["validation"]
    modelPath = args["model_name"]
    plotPath = args["plot"]

    print("Preprocessing the raw datesets!")

    print("Build feature vectors!")

    print("Construct model!")

    print("Summary of the constructed model!")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

    print("Train model!")

    print("Save model, weights and plot!")

    print("Predict with model!")



if __name__ == "__main__":
    main()
