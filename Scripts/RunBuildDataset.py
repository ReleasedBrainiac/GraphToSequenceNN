from DatasetHandler.DatasetProvider import DatasetPipeline

#===========================================================#
#                            Test                           #
#===========================================================#
inpath = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
pipe = DatasetPipeline(in_path=inpath, 
                       output_path_extender='dc.ouput', 
                       max_length=-1, 
                       show_feedback=True, 
                       saving=False, 
                       keep_edges=False
                       )

pipe.ProvideData()