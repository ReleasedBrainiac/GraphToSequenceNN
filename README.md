# GraphToSequenceNN [In Progress] 

This Repo provides a Graph2Sequnce for Keras implementation. It will provide an example of Graph2Sequence encoding and decoding. The idea behind the Graph2Sequence is based on https://arxiv.org/abs/1804.00823. In this implementation I try to encode the complex datasets from https://amr.isi.edu/ (Abstract Meaning Representation). 

The IBM Research Team implemted a tensorflow Graph2Sequence model with attention => https://github.com/IBM/Graph2Seq !
There are hughe differences between our models structures and used functions and implementations.
This is caused by the differnt interpretation of the paper and my goal to strictly implement the paper as much as possible.

## The Setup (necessary libraries!)

- Python version:       3.5.6 => https://www.python.org/downloads/release/python-365/
- Tensorflow Version:   1.10.0 => https://www.tensorflow.org/ [For GPU or CPU]
- Keras Version:        2.2.2 => https://keras.io/
- Anytree Version:      2.4.3 => https://pypi.org/project/anytree/
- Ordereddict Version:  1.1   => https://pypi.org/project/ordereddict/
- Pydot Version:        1.2.4 => https://pypi.org/project/pydot/
- Graphviz Version:     0.8.4 => https://pypi.org/project/graphviz/
- H5PY                        => http://docs.h5py.org/en/latest/build.html
- Matplotlib            1.1.0 => https://matplotlib.org/faq/installing_faq.html

## Usage

### Execution:
1. download and unpack the 'The Little Prince' corpus from https://amr.isi.edu/download.html. 
2. change the global values in the "main.py" as you desired (at least provide the dataset path)
3. open you desired command line, powershell or bash
4. execute "python your_path_to/main.py"
5. see the magic

### Attention: 
- Change "STORE_STDOUT" to False to see console reports... otherwise they gonna be stored in a file and the console will be empty until the process is done!

## Current developement status

- Tool is Finished

## Resources

- I used the following ressource for lookup the table replacement.
        => https://www.thesaurus.com/browse/clean%20out?s=t 
- Try Catch Style/Code Source:
        => https://stackoverflow.com/questions/9823936/python-how-do-i-know-what-type-of-exception-occurred

## Tools

- Live Regex Online Tool => https://regex101.com/r/U7SV1y/1/ 
- Visual Studio Code + Extensions

## Questions
- [?]  Is GloVe able to encode "New Zealand", "North_Pole" or other definitions

## [AMR Parser] What do I expel from AMR string
 
1. Replaced all base word extensions 
        1. "do-01" => "d0"
2. Replaced all not alphanumeric qualified strings "" 
        1. "Ocean" => kept
        2. "19:35" => expelled
3. Replaced polaritity "-" with a new node 
        1. => {"N0T": "not"}
4. Replaced words with a look up listcontaining words with a close meaning
        1. "littly-by-little" => "gradually"
        2. "amr-unknown" => "?"
5. Replaced all options ":opx 'Zealand'" with a new node 
        1. => {"NSLx": "Zealand"}
6. Expelled all signs which not match alpahnumeric, whitespacing, "/" and round parenthesis

## [GloVe Dataset Parser]

1. Removed ['#::snt ', '" ',' "'] and replaced '- -' with '-' in the sentences

[THE RESULT] is a cleaned AMR string which i gonna use for to create a node embedding with GloVe.

## [Bugs]

1. It isn't possible to extend the batch_size > 1 [Fix status => PAUSED]
2. For python 3.6.5 saving the whole model throws an key argument error [Fix status => ignored]
        1. Currently saving weights is possible
                                          
