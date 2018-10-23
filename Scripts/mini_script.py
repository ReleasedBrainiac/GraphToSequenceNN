import re
from DatasetHandler.FileReader import Reader

path = ''
content = Reader(path)

content = re.sub(', ', '\n', content)   