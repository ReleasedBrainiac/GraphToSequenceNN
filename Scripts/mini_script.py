from DatasetHandler.FileReader import Reader
from DatasetHandler.FileWriter import Writer

#path = '../Datasets/supported_amr_internal_nodes.txt'
#content =  str(Reader(input_path=path).Read()).replace(', ', ',\n').replace('\'', '').replace(': ', '#').replace('["', '').replace('"]', '')
#Writer(input_path=path, in_context=content).StoreContext()

look_up_extension_replace_path = '../Datasets/LookUpAMR/supported_amr_internal_nodes_lookup.txt'
content =  str(Reader(input_path=look_up_extension_replace_path).LineReadContent())
print(content)