# - *- coding: utf- 8*-
#This script was created by T.Turke for content extraction from AMR dataset.
#~> https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
#~> https://www.pythonsheets.com/notes/python-rexp.html

import re
from TextFormatting.Contentsupport import *
from GraphHandler.Graphsalvage import Gatherer

#===========================================================#
#                   Methods/Functions                       #
#===========================================================#

#==         Calculate MW for a list of numbers            ==#
# This function calculate the mean over all values in a list.
def CalcMW(sentence_length):
    if(isList(sentence_length)):
        sent_summ = 0

        for index, keys in enumerate(sentence_length):
            sent_summ += sentence_length[index]

        mw = int(round(sent_summ / len(sentence_length)))
        return mw
    else:
        print('WRONG INPUT FOR [CalcMW]')
        return None

#==                    Read AMR Dataset                   ==#
# This function provide a file reader for the AMR dataset.
def FileToString(xpath):
    if(isStr(xpath)):
        with open(xpath, 'r', encoding="utf8") as fileIn:
            data=fileIn.read()
            result=data.split('#')
            return result
    else:
        print('WRONG INPUT FOR [FileToString]')
        return None

#==                    Restrict Content                   ==#
# This funtion check a sentence and semantic pair satisfy the size restiction and return 2 values:
# 1. boolean to validate the check
# 2. concatenation of sentence and semantic
def ValidateAndCreateWriteCorpus(max_len, sent, sem, sen_size, sem_size):
    if(isInt(max_len)) and (isInt(sen_size)) and (isInt(sem_size)) and (isStr(sent)) and (isNotNone(sem)):
        if (max_len < 1) or ((max_len > 0) and (len(sent) < (sen_size+1)) and (len(sem) < (sem_size+1))):
            return [True, sent + sem]
    else:
        print('WRONG INPUT FOR [CreateWriteCorpus]')
        return [False, sent + sem]

#==                     Filter Content                    ==#
# This method clean up the a given amr extracted sentence from text formating markup.
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
# Depending on the flag:
# 1. print_console    => If True you get console output for Graphsalvage.Gatherer
# 2. to_process       => If True you get GraphTree as AnyTree for further usage
#                     => If False you get a JsonString for saving in file.
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
# Additional we have the following options:
# 1. want_as_arm    => If True then it return a tree-like formated AMR string for the semantic entry 
#                   => ATTENTION: this option does not support conversion with ConvertToTensorMatrices!
# 2. isShowConsole  => If True then it show in and out of the Graphsalvage.Gatherer on console
# 3. isNotStoring   => If True you get GraphTree as AnyTree for further usage.
#                   => If False you get a JsonString for saving in file.
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
def GetMultiDatasetPairs(sent_flag, sem_flag, sent_array, sem_array, want_as_arm, isShowConsole, isNotStoring):
    if (isStr(sent_flag)) and (isStr(sem_flag)) and (isList(sent_array)) and (isList(sem_array)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
        dataset_pairs_sent_sem = []
        for i in range(min(len(sent_array), len(sem_array))):
            dataset_pairs_sent_sem.append(GetSingleDatasetPair(sent_flag, sem_flag, sent_array[i], sem_array[i], want_as_arm, isShowConsole, isNotStoring))

        return dataset_pairs_sent_sem
    else:
        print('WRONG INPUT FOR [GetMultiDatasetPair]')
        return []

def GetAnyTreeDataset():
    for i in range(min(len(sent_array), len(sem_array))):
            #Gather sentences without linking substrings because they are not contained in the semantic graph
            sent = sent_flag+' '+sent_array[i]
            sent = ClearSentence(sent)
            sem = ReforgeSemanticRepresentation(sem_array[i], sem_flag);

#==        Extract sentence element from AMR input        ==#
# This function extract the sentence element from AMR corpus element!
def ExtractSentence(x_delim, in_content, index):
    raw_start_index = in_content[index].find(x_delim)+6
    sentence = in_content[index]
    sent_len = len(sentence)
    return sentence[raw_start_index:sent_len-1]

#==        Extract semantics element from AMR input        ==#
# This function extract the semantics element from AMR corpus element!
def ExtractSemantics(in_content, index):
    raw_content = in_content[index].split('.txt')
    raw_con_index = len(raw_content)-1
    return raw_content[raw_con_index]

#==             Record clean_content extractor            ==#
# This function collect the AMR-String-Representation and the corresponding sentence from AMR corpus.
# Returns:
# 1. list of length of each sentence
# 2. list of length of each semantic (= AMR-String-Representation) 
# 3. list of sentences
# 4. list of semantics (= AMR-String-Representation) 
# => 1., 2., 3. and 4. should be equal in length!
def ExtractContent(in_content, x_delim, y_delim):
    if isNotNone(in_content) and isStr(x_delim) and isStr(y_delim):
        sent_lens = []
        sem_lens = []
        sentences = []
        semantics = []

        for index, elem in enumerate(in_content):
            if x_delim in elem:
                sentence = ExtractSentence(x_delim, in_content, index)
                sent_lens.append(len(sentence))
                sentences.append(sentence)

            if y_delim in elem:
                semantic = ExtractSemantics(in_content, index)
                sem_lens.append(len(semantic))
                semantics.append(semantic)

        return [sent_lens, sem_lens, sentences, semantics]
    else:
        print('WRONG INPUT FOR [ExtractContent]')
        return None

#==             Network dataset preprocessing             ==#
def ConvertToTensorMatrices():
    print('This function is work in progress! [ConvertToTensorMatrices]')
    return None

#===========================================================#
#                          Storing                          #
#===========================================================#

#==                    Write AMR Dataset                  ==#
# This function 
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
#                          Execution                        #
#===========================================================#

#                          Pipeline                         #
# This function 
def pipeline(inpath, output_extender, max_length, save_as_arm, print_console):
    #==                       Variables                       ==#
    semantics  = []
    sentences  = []
    sents_lens = []
    sema_lens  = []

    typerror = 'Entered wrong type!'
    sentence_delim = '::snt'
    semantik_delim = '::smt'
    file_delim = '::file'

    max_length = setOrDefault(max_length, -1, isInt(max_length));
    inpath  = setOrDefault(inpath, typerror, isStr(inpath));
    outpath = setOrDefault(inpath+'.'+ output_extender , typerror, isStr(output_extender));

    print('max_length: ', max_length)
    print('inpath: ', inpath)
    print('outpath: ', outpath)

    #==                     Read Dataset                      ==#
    result = FileToString(inpath)

    #==             Collect relevant raw_content              ==#
    len_dataset = len(result)
    result=result[1:len_dataset]
    sents_lens, sema_lens, sentences, semantics = ExtractContent(result, sentence_delim, file_delim)

    #==                     Check Median                      ==#
    mw_value_sen = CalcMW(sents_lens)
    mw_value_sem = CalcMW(sema_lens)
    print(mw_value_sen)
    print(mw_value_sem)

    #==                 Sava Collected Content                ==#

    data_pairs = GetMultiDatasetPairs(sentence_delim, semantik_delim, sentences, semantics, save_as_arm, print_console, True)

    SaveToFile(outpath,
               mw_value_sen,
               mw_value_sem,
               max_length,
               data_pairs)
    
    return None
