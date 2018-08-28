from TextFormatting.DatasetProvider import SavePipeline as Pipe

#===========================================================#
#                            Test                           #
#===========================================================#
max_length = -1
output_extender = 'dc.ouput'
inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
save_as_arm = False
print_activated_anynode = True  # this wont work if you only store arm cause no anynode is calced!

Pipe(inpath, output_extender, max_length, save_as_arm, print_activated_anynode)
