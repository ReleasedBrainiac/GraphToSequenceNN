from DatasetHandler.DatasetProvider import DatasetPipelines

#===========================================================#
#                            Test                           #
#===========================================================#
max_length = -1
output_extender = '.ouput'
#inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
inpath = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'
save_amr = False
print_activated = True  # this wont work if you only store arm cause no anynode is calced!

pipe = DatasetPipelines(in_path=inpath, output_path_extender=output_extender, max_length=max_length, show_feedback=print_activated, as_amr=save_amr)
pipe.SaveData()