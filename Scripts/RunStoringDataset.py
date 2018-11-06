from DatasetHandler.DatasetProvider import DatasetPipelines

#===========================================================#
#                            Test                           #
#===========================================================#
max_length = -1
output_extender = '.ouput'
#inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
inpath = './Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-training.txt'

as_amr = True
saving = True
print_activated = False  # this wont work if you only store arm cause no anynode is calced!

pipe = DatasetPipelines(in_path=inpath, output_path_extender=output_extender, max_length=max_length, show_feedback=print_activated, saving=saving)
pipe.SaveData(as_amr=as_amr)