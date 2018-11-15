from DatasetHandler.DatasetProvider import DatasetPipelines

#===========================================================#
#                            Test                           #
#===========================================================#
#inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
#inpath = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
inpath = './Datasets/Raw/AMR Bio/amr-release-dev-bio.txt'

# show_feedback = True => this wont work if you only store arm cause no anynode is calced!

pipe = DatasetPipelines(in_path=inpath, 
                        output_path_extender='ouput', 
                        max_length=-1, 
                        show_feedback=True, 
                        saving=True, 
                        keep_edges=True
                        )

pipe.SaveData(as_amr=True)