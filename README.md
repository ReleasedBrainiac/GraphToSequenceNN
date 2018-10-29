# GraphToSequenceNN [In Progress] 

This Repo is in proceeding. It will provide an example of Graph2Sequence encoding and decoding. The source behind the Graph2Sequence idea is https://arxiv.org/abs/1804.00823. In this implementation I try to encode datasets from https://amr.isi.edu/ (Abstract Meaning Representation). 

## The Setup (necessary libraries!)

- Tensorflow Version:   1.10.0 => https://www.tensorflow.org/ [For GPU]
- Keras Version:        2.2.2 => https://keras.io/
- Python version:       3.5.6 => https://www.python.org/downloads/release/python-365/
- Anytree Version:      2.4.3 => https://pypi.org/project/anytree/
- NetworkX              2.2   => https://networkx.github.io/documentation/stable/install.html
- Ordereddict Version:  1.1   => https://pypi.org/project/ordereddict/
- Pydot Version:        1.2.4 => https://pypi.org/project/pydot/
- Graphviz Version:     0.8.4 => https://pypi.org/project/graphviz/
- H5PY                        => http://docs.h5py.org/en/latest/build.html
- Jupyter Notebook [optional] => http://jupyter.org/install 

## Current developement status

- Currently the programm is not working. I'ts still at the beginning!

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
4. Replaced word lists like with word with a close meaning
        1. "littly-by-little" => "gradually"
        2. "amr-unknown" => "?"
5. Replaced all options ":opx 'Zealand'" with a new node 
        1. => {"NSLx": "Zealand"}
6. Expelled all signs which not match alpahnumeric, whitespacing, "/" and round parenthesis

[THE RESULT] is a cleaned AMR string which i gonna use for to create a node embedding with GloVe.
                                          