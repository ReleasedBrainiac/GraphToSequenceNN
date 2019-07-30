# GraphToSequenceNN 

This Repo provides a Graph2Sequnce for Keras implementation. It will provide an example of Graph2Sequence encoding and decoding. The idea behind the Graph2Sequence is based on https://arxiv.org/abs/1804.00823. In this implementation I try to encode the complex datasets from https://amr.isi.edu/ (Abstract Meaning Representation). 

The IBM Research Team implemted a tensorflow Graph2Sequence model with attention => https://github.com/IBM/Graph2Seq !

There are differences between our models structures and used functions and implementations.
This is caused by the differnt interpretation of the paper.

## The Setup (necessary libraries!)

- Python version:       3.6.5 => https://www.python.org/downloads/release/python-365/ (v.3.6.5 and above)
- Tensorflow Version:  1.12.0 => https://www.tensorflow.org/ [For GPU or CPU] (v1.12. and above)
- Keras Version:        2.2.2 => https://keras.io/
- Anytree Version:      2.4.3 => https://pypi.org/project/anytree/
- Ordereddict Version:  1.1   => https://pypi.org/project/ordereddict/
- Pydot Version:        1.2.4 => https://pypi.org/project/pydot/
- Graphviz Version:     0.8.4 => https://pypi.org/project/graphviz/
- H5PY                        => http://docs.h5py.org/en/latest/build.html
- Matplotlib            1.1.0 => https://matplotlib.org/faq/installing_faq.html
- Sklearn                     => https://scikit-learn.org/stable/install.html

## Usage

1. download and unpack the 'The Little Prince' corpus from https://amr.isi.edu/download.html. 
2. download and unpack your desired GloVe pretrained word vectors https://nlp.stanford.edu/projects/glove/
3. change the global values in the "main.py" as you desired (at least provide the dataset and glove path)
4. open you desired command line, terminal, powershell, bash or etc.
5. execute "python your_path_to/main.py"
6. see the magic

Additional: A much bigger Dataset [https://github.com/freesunshine0316/neural-graph-to-seq-mp] provided in Yue Zhang et al. => [https://frcchang.github.io/pub/acl18.song.pdf] 

### Attention: 
- The console output will be logged completely and also shown. SET model.fit param verbose to 0 to reduce the log file size! 

## Developement 

### Status
- Tool is nearly Finished

### Resources
- I used the following ressource for lookup the table replacement.
        => https://www.thesaurus.com/browse/clean%20out?s=t 
- Try Catch Style/Code Source:
        => https://stackoverflow.com/questions/9823936/python-how-do-i-know-what-type-of-exception-occurred

- Additional resource can be found in the code files headings! (To much for the Readme...)

### Tools
- Live Regex Online Tool => https://regex101.com/r/U7SV1y/1/ 
- Visual Studio Code + Extensions

### Questions
- [?]  Is GloVe able to encode "New Zealand", "North_Pole" or other definitions

## [AMR Parser] What's expeled from AMR string?
 
Replaced all base word extensions 
  1. "do-01" => "do"

Replaced all not alphanumeric qualified strings "" 
  1. "Ocean" => kept
  2. "19:35" => expelled

Replaced polaritity "-" with a new node 
  1. {"N0T": "not"}

Replaced words with a look up listcontaining words with a close meaning
  1. "littly-by-little" => "gradually"
  2. "amr-unknown" => "?"

Replaced all options ":opx 'Zealand'" with a new node 
  1. {"NSLx": "Zealand"}

Expelled all signs which not match alpahnumeric, whitespacing, "/" and round parenthesis


## [GloVe Dataset Parser] What's changed for the process?
In the sentences:
  1. Removed ['#::snt ', '" ',' "'] 
  2. Replaced '- -' with '-'

[THE RESULT] is a cleaned AMR string which i gonna use for to create a node embedding with GloVe.

## [Bugs]

It isn't possible to extend the batch_size > 1 
  * [Fix status => PAUSED]

For python 3.6.5 saving the whole model throws an key argument error 
  * [Fix status => IGNORED]
  * [POSSIBLE] Saving weights
                                          
