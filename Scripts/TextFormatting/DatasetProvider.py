# - *- coding: utf- 8*-
#This script was created by T.Turke for content extraction from AMR dataset.
#~> https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
#~> https://www.pythonsheets.com/notes/python-rexp.html

import re
from TextFormatting.ContentSupport import *
from GraphHandler.GraphSalvage import Gatherer

#===========================================================#
#                   Methods/Functions                       #
#===========================================================#

#==         Calculate MW for a list of numbers            ==#
# This function calculate the mean over all values in a list.
#
#   Inputs:
#       sentences_length => lengths of all sentences
#
#   Return:
#       The mean over all sentences
def CalcMW(sentences_length):
    if(isList(sentences_length)):
        sent_summ = 0

        for index, keys in enumerate(sentences_length):
            sent_summ += sentences_length[index]

        mw = int(round(sent_summ / len(sentences_length)))
        return mw
    else:
        print('WRONG INPUT FOR [CalcMW]')
        return None

#==                    Read AMR Dataset                   ==#
# This function provide a file reader for the AMR dataset.
#
#   Inputs:
#       path    => path string to dataset text file
#
#   Return:
#       The dataset string.
def FileToString(path):
    if(isStr(path)):
        with open(path, 'r', encoding="utf8") as fileIn:
            data=fileIn.read()
            content=data.split('#')
            return content
    else:
        print('WRONG INPUT FOR [FileToString]')
        return None

#==                                  Restrict Content                            ==#
# This funtion check a sentence and semantic pair satisfy the size restictions.
#
#   Inputs:
#       max_len     => max allow length of a sentence
#       sent        => the cleaned sentence
#       sem         => the cleaned correspondign semantic for the sentence
#       sen_size    => allowed size for sentences
#       sem_size    => allowed size for semantics
#
#   Return:
#       A list of a validation boolean and the string concatenation of semantic and sentece.
def ValidateAndCreateWriteCorpus(max_len, sent, sem, sen_size, sem_size):
    if(isInt(max_len)) and (isInt(sen_size)) and (isInt(sem_size)) and (isStr(sent)) and (isNotNone(sem)):
        if (max_len < 1) or ((max_len > 0) and (len(sent) < (sen_size+1)) and (len(sem) < (sem_size+1))):
            return [True, sent + sem]
    else:
        print('WRONG INPUT FOR [CreateWriteCorpus]')
        return [False, sent + sem]

#==                                Filter Content                                ==#
# This method clean up the a given amr extracted sentence from text formating markup.
#
#   Inputs:
#       in_sentence => raw sentence AMR split element 
#
#   Returns:
#       The cleaned sentence.
def ClearSentence(in_sentence):
    if(isStr(in_sentence)):
        in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
        in_sentence = re.sub('</[^>]+>','', in_sentence)
        in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
        return in_sentence+'\n'
    else:
        print('WRONG INPUT FOR [ClearSentence]')
        return in_sentence

#==                Reforge AMR semantic to cleaned AMR string tree               ==#
# This function allow to clean up a raw AMR semantic string tree representation 
# into a cleaned version of it.
#
#   Inputs:
#       semantic    => raw semantic input
#       sem_flag    => marker/delim to add to cleaned semantic
#
#   Returns:
#       A tree formated string like AMR.
def ReforgeSemanticRepresentationToCleanARM(semantic, sem_flag):
    if(isStr(semantic)) and (isStr(sem_flag)):
        half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
        out = half_cleaned_sem+'\n'
        return out
    else:
        print('WRONG INPUT FOR [ReforgeSemanticRepresentationToCleanARM]')
        return in_sentence

#==                Reforge AMR semantic to cleaned AnyTree object                ==#
# This function allow to clean up a raw AMR semantic string tree representation 
# into a cleaned anytree reprenstation of it.
#   Inputs:
#       semantic    => raw semantic input
#       sem_flag    => marker/delim to add to cleaned semantic
#
#   Options:
#       print_console    => If True you get console output for GraphSalvage.Gatherer
#       to_process       => If True you get GraphTree as AnyTree for further usage
#                        => If False you get a JsonString for saving in file.
#
#   Returns:
#       A reforged semantic as AnyNode object.
def ReforgeSemanticRepresentationToAnyTree(semantic, sem_flag, print_console, to_process):
    if(isStr(semantic)) and (isStr(sem_flag)) and (isBool(print_console)) and (isBool(to_process)):
        half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
        out = half_cleaned_sem+'\n'
        return Gatherer(out, sem_flag, print_console, to_process)
    else:
        print('WRONG INPUT FOR [ReforgeSemanticRepresentationToAnyTree]')
        return in_sentence

#==                       Collect single dataset data pair                       ==#
# This function collect a data pair from raw sentences and semantics.
#   Inputs:
#       sent_flag    => defines a marker to attach to the cleaned sentence, make it easier to find later
#       sem_flag     => defines a marker to attach to the cleaned semantic, make it easier to find later
#       sent         => is the raw input of the AMR sentence
#       sem          => is the raw input of the AMR semantic
#
#   Additional we have the following options:
#       want_as_arm     => If True then it return a tree-like formated AMR string for the semantic entry 
#                       => ATTENTION: this option does not support conversion with ConvertToTensorMatrices!
#       isShowConsole   => If True then it show in and out of the GraphSalvage.Gatherer on console
#       isNotStoring    => If True you get GraphTree as AnyTree for further usage.
#                       => If False you get a JsonString for saving in file.
#
#   Returns:
#       A single data pair like [sentence, semantic]
def GetSingleDatasetPair(sent_flag, sem_flag, sent, sem, want_as_arm, isShowConsole, isNotStoring):
    if (isStr(sent_flag)) and (isStr(sem_flag)) and (isStr(sent)) and (isStr(sem)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
        sent = sent_flag+' '+sent
        sent = ClearSentence(sent)
        if(want_as_arm):
            sem = ReforgeSemanticRepresentationToCleanARM(sem, sem_flag);
            return [sent, sem]
        else:
            sem = ReforgeSemanticRepresentationToAnyTree(sem, sem_flag, isShowConsole, isNotStoring);
            return [sent, sem]
    else:
        print('WRONG INPUT FOR [GetSingleDatasetPair]')
        return [None, None]

#==                       Collect multi dataset data pair                       ==#
# This function collect multiples pairs of semantic and sentence data as list of data pairs.
# For this case we pass arrays of raw sentences and semantics where index i in both arrays point to a sentence and the corresponding semantic.
#   Inputs: 
#       sent_array   => Array of the raw input of the AMR sentences
#       sem_array    => Array of the raw input of the AMR semantics
#
#       The other variables are same as defined in function GetSingleDatasetPair.
#
#   Returns:
#       A list of data pairs results from GetSingleDatasetPair
def GetMultiDatasetPairs(sent_flag, sem_flag, sent_array, sem_array, want_as_arm, isShowConsole, isNotStoring):
    if (isStr(sent_flag)) and (isStr(sem_flag)) and (isList(sent_array)) and (isList(sem_array)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
        dataset_pairs_sent_sem = []
        for i in range(min(len(sent_array), len(sem_array))):
            dataset_pairs_sent_sem.append(GetSingleDatasetPair(sent_flag, sem_flag, sent_array[i], sem_array[i], want_as_arm, isShowConsole, isNotStoring))

        return dataset_pairs_sent_sem
    else:
        print('WRONG INPUT FOR [GetMultiDatasetPair]')
        return []

#==        Extract sentence element from AMR input        ==#
# This function extract the sentence element from AMR corpus element!
#   Inputs:
#       x_delim     =>  marker/delim to find the sentence fragment
#       in_content  =>  raw amr string fragment from split of full AMR dataset string
#       index       =>  position where the sentence was found
#
#   Returns:
#       The extracted raw sentence at the given position.
def ExtractSentence(x_delim, in_content, index):
    raw_start_index = in_content[index].find(x_delim)+6
    sentence = in_content[index]
    sent_len = len(sentence)
    return sentence[raw_start_index:sent_len-1]

#==        Extract semantics element from AMR input        ==#
# This function extract the semantics element from AMR corpus element!
#   Inputs:
#       in_content  =>  raw amr string fragment from split of full AMR dataset string
#       index       =>  position where the semantic was found
#
#   Returns:
#       The extracted raw semantic at the given position.
def ExtractSemantics(in_content, index):
    raw_content = in_content[index].split('.txt')
    raw_con_index = len(raw_content)-1
    return raw_content[raw_con_index]

#==             Record clean_content extractor            ==#
# This function collect the AMR-String-Representation and the corresponding sentence from AMR corpus.
#   Inputs:
#       in_content  =>  raw amr string fragment from split of full AMR dataset string
#       x_delim     =>  marker/delim to validate fragment as raw sentence
#       y_delim     =>  marker/delim to validate fragment as raw semantic
#
#   Returns:
#       sent_lens   => list of length of each sentence
#       sem_lens    => list of length of each semantic (= AMR-String-Representation) 
#       sentences   => list of sentences
#       semantics   => list of semantics (= AMR-String-Representation) 
#       => all 4. should be equal in length!
def ExtractContent(in_content, x_delim, y_delim):
    if isNotNone(in_content) and isStr(x_delim) and isStr(y_delim):
        sentence = ''
        semantic = ''
        sentence_found = False
        semantic_found = False
        sent_lens = []
        sem_lens = []
        sentences = []
        semantics = []

        for index, elem in enumerate(in_content):
            if (x_delim in elem) and (not sentence_found):
                sentence = ExtractSentence(x_delim, in_content, index)
                sentence_found = True

            if (y_delim in elem)  and (not semantic_found):
                semantic = ExtractSemantics(in_content, index)
                semantic_found = True

            if sentence_found and semantic_found:
                sent_lens.append(len(sentence))
                sem_lens.append(len(semantic))
                sentences.append(sentence)
                semantics.append(semantic)
                sentence_found = False
                semantic_found = False

        if(len(sent_lens) == len(sem_lens) == len(sentences) == len(semantics)):
            return [sent_lens, sem_lens, sentences, semantics]
        else:
            print('WRONG OUTPUT FOR [ExtractContent]... Size of outputs dont match!')
            return None
    else:
        print('WRONG INPUT FOR [ExtractContent]')
        return None

#==             Network dataset preprocessing             ==#
# This function 
#
#   Inputs:
#       data_pairs  => list of data pairs from GetSingleDatasetPair function
#
#   Return:
#      
def ConvertToTensorMatrices(data_pairs):
    print('This function is work in progress! [ConvertToTensorMatrices]')
    return None

#===========================================================#
#                          Storing                          #
#===========================================================#

#==                    Write AMR Dataset                  ==#
# This function 
#
#   Inputs:
#       path        => path to output file 
#       len_sen_mw  => mean of sentences length
#       len_sem_mw  => mean of semantics length
#       max_len     => max length of sentences we desire to store
#       data_pairs  => result data pairs as list
#       
def SaveToFile(path, len_sen_mw, len_sem_mw, max_len, data_pairs):
     with open(path, 'w', encoding="utf8") as fileOut:
        sen_size = min(len_sen_mw, max_len)
        sem_size = min(len_sem_mw, (max_len*2))

        for i in range(len(data_pairs)):
            #Restrict writing content
            isAllowed, out = ValidateAndCreateWriteCorpus(max_len, data_pairs[i][0], data_pairs[i][1], sen_size, sem_size)
            if (isAllowed):
                fileOut.write(out)
                fileOut.flush()

        print(path)
        return None

#===========================================================#
#                  End-to-End Saving-Execution              #
#===========================================================#

#                       Save Pipeline                       #
# This function 
#
#   Inputs:
#       inpath          => path of dataset text file
#       output_extender => extender to define result filename
#       max_length      => max allow length for sentences
#
#   Options:
#       save_as_arm     => output will be save as tree like formated AMR string
#       print_console   => show all Reforging at the Gatherer on console
#       
def SavePipeline(inpath, output_extender, max_length, save_as_arm, print_console):
    semantics  = []
    sentences  = []
    sents_lens = []
    sema_lens  = []

    # Constants for the AMR processing
    TYP_ERROR = 'Entered wrong type!'
    SENTENCE_DELIM = '::snt'
    SEMANTIC_DELIM = '::smt'
    FILE_DELIM = '::file'

    max_length = setOrDefault(max_length, -1, isInt(max_length));
    inpath  = setOrDefault(inpath, TYP_ERROR, isStr(inpath));
    outpath = setOrDefault(inpath+'.'+ output_extender , TYP_ERROR, isStr(output_extender));

    print('max_length: ', max_length)
    print('inpath: ', inpath)
    print('outpath: ', outpath)

    #==                     Read Dataset                      ==#
    dataset = FileToString(inpath)

    #==             Collect relevant raw_content              ==#
    len_dataset = len(dataset)
    dataset=dataset[1:len_dataset]
    sents_lens, sema_lens, sentences, semantics = ExtractContent(dataset, SENTENCE_DELIM, FILE_DELIM)

    #==                      Get Median                       ==#
    mw_value_sen = CalcMW(sents_lens)
    mw_value_sem = CalcMW(sema_lens)
    print('Mean sentences: ', mw_value_sen)
    print('Mean semantics: ', mw_value_sem)

    #==                 Sava Collected Content                ==#

    data_pairs = GetMultiDatasetPairs(SENTENCE_DELIM, SEMANTIC_DELIM, sentences, semantics, save_as_arm, print_console, True)

    SaveToFile(outpath,
               mw_value_sen,
               mw_value_sem,
               max_length,
               data_pairs)
    
    return None
