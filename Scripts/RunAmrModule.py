from DatasetHandler.DatasetProvider import DatasetPipelines as Pipe

#===========================================================#
#                            Test                           #
#===========================================================#
max_length = -1
output_extender = 'dc.ouput'
inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
print_activated_anynode = False  # this wont work if you only store arm cause no anynode is calced!
is_not_saving = True

Pipe().BasicPipeline(inpath, output_extender, max_length, False, print_activated_anynode, is_not_saving)