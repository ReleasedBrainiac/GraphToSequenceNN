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
def CreateWriteCorpus(max_len, sent, sem, sen_size, sem_size):
    if (max_len < 1) or ((max_len > 0) and (len(sent) < (sen_size+1)) and (len(sem) < (sem_size+1))):
        return [True, sent + sem]

#==                     Filter Content                    ==#
def ClearSentence(in_sentence):
    in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
    in_sentence = re.sub('</[^>]+>','', in_sentence)
    in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
    return in_sentence+'\n'

def ReforgeSemanticRepresentationToCleanARM(semantic, sem_flag):
    half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
    out = half_cleaned_sem+'\n'
    return out

def ReforgeSemanticRepresentationToAnyTree(semantic, sem_flag, print_activated, to_process):
    half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
    out = half_cleaned_sem+'\n'
    return Gatherer(out, sem_flag, print_activated, to_process)

#==                    Write AMR Dataset                  ==#
def SaveToFile(ypath, sent_array, sem_array, sent_flag, sem_flag, len_sen_mw, len_sem_mw, max_len, want_arm, print_activated):
    with open(ypath, 'w', encoding="utf8") as fileOut:
        sen_size = min(len_sen_mw, max_len)
        sem_size = min(len_sem_mw, (max_len*2))

        for i in range(min(len(sent_array), len(sem_array))):
            #Gather sentences without linking substrings because they are not contained in the semantic graph
            sent = sent_flag+' '+sent_array[i]
            sent = ClearSentence(sent)
            if(want_arm):
                sem = ReforgeSemanticRepresentationToCleanARM(sem_array[i], sem_flag);
            else:
                sem = ReforgeSemanticRepresentationToAnyTree(sem_array[i], sem_flag, print_activated, True);
            
            #Restrict writing content
            isAllowed, out = CreateWriteCorpus(max_len, sent, sem, sen_size, sem_size)
            if (isAllowed):
                fileOut.write(out)
                fileOut.flush()
                
    print(ypath)
    return None

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

#===========================================================#
#                          Pipeline                         #
#===========================================================#
def pipeline(inpath, output_extender, max_length_sentences, save_as_arm, print_activated):
    #==                       Variables                       ==#
    semantics  = []
    sentences  = []
    sents_lens = []
    sema_lens  = []

    typerror = 'Entered wrong type!'
    sentence_delim = '::snt'
    semantik_delim = '::smt'
    file_delim = '::file'

    max_length = setOrDefault(max_length_sentences, -1, isInt(max_length_sentences));
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
    SaveToFile(outpath,
               sentences,
               semantics,
               sentence_delim,
               semantik_delim,
               mw_value_sen,
               mw_value_sem,
               max_length,
               save_as_arm,
               print_activated)
    
    return None
