from TextFormatting.Datasetprovider import pipeline as pipe

#===========================================================#
#                            Test                           #
#===========================================================#
max_length = -1
output_extender = 'dc.ouput'
inpath = '../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'


pipe(inpath, output_extender, max_length)
