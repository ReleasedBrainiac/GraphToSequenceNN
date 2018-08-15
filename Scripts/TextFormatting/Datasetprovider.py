# - *- coding: utf- 8*-
#This script was created by T.Turke for content extraction from AMR dataset.
#~> https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
#~> https://www.pythonsheets.com/notes/python-rexp.html

import re
from TextFormatting.Contentsupport import setOrDefault as sod
from TextFormatting.Contentsupport import isStr, isInt
from GraphHandler.Graphsalvage import GatherGraphInfo

#===========================================================#
#                   Methods/Functions                       #
#===========================================================#

#==         Calculate MW for a list of numbers            ==#
def CalcMW(sentence_length):
    sent_summ = 0

    for index, keys in enumerate(sentence_length):
        sent_summ += sentence_length[index]

    mw = int(round(sent_summ / len(sentence_length)))
    return mw

#==                    Read AMR Dataset                   ==#
def FileToString(xpath):
    with open(xpath, 'r', encoding="utf8") as fileIn:
        data=fileIn.read()
        result=data.split('#')
        return result

#==                    Restrict Content                   ==#
def CreateWriteCorpus(max_len, sent, sem):
    if (max_len < 1) or ((max_len > 0) and (len(sent) < (allowed_size_sen+1)) and (len(sem) < (allowed_size_sem+1))):
        return [True, sent + sem]

#==                     Filter Content                    ==#
def ClearSentence(in_sentence):
    in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
    in_sentence = re.sub('</[^>]+>','', in_sentence)
    in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
    return in_sentence+'\n'

def ReforgeSemanticRepresentation(semantic, sem_flag):
    #half_cleaned_sem = '#'+sem_flag+' '+semantic.replace('\n', '|')+'\n'
    half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
    #return re.sub(' +',' ',half_cleaned_sem)+'\n'
    out = half_cleaned_sem+'\n'
    GatherGraphInfo(out, sem_flag)
    return out

#==                    Write AMR Dataset                  ==#
def StringToFile(ypath, sent_array, sem_array, sent_flag, sem_flag, len_sen_mw, len_sem_mw, max_len):
    with open(ypath, 'w', encoding="utf8") as fileOut:
        allowed_size_sen = min(len_sen_mw, max_len)
        allowed_size_sem = min(len_sem_mw, (max_len*2))

        for i in range(min(len(sent_array), len(sem_array))):
            #Gather sentences without linking substrings because they are not contained in the semantic graph
            sent = sent_flag+' '+sent_array[i]
            sent = ClearSentence(sent)
            sem = ReforgeSemanticRepresentation(sem_array[i], sem_flag);
            #Restrict writing content
            isAllowed, out = CreateWriteCorpus(max_len, sent, sem)
'''
            if (isAllowed):
                fileOut.write(out)
                fileOut.flush()
    return None
'''

#==             Record clean_content extractor            ==#
def ExtractContent(in_content, x_delim, y_delim):
    sent_lens = []
    sem_lens = []
    sentences = []
    semantics = []

    for index, elem in enumerate(in_content):
        if x_delim in elem:
            raw_start_index = in_content[index].find(x_delim)+6
            sentence = in_content[index]
            sent_len = len(sentence)
            sentence = sentence[raw_start_index:sent_len-1]
            sent_lens.append(len(sentence))
            sentences.append(sentence)

        if y_delim in elem:
            raw_content = in_content[index].split('.txt')
            raw_con_index = len(raw_content)-1
            semantic = raw_content[raw_con_index]
            sem_lens.append(len(semantic))
            semantics.append(semantic)

    return [sent_lens, sem_lens, sentences, semantics]

#===========================================================#
#                          Pipeline                         #
#===========================================================#
def pipeline(inpath, output_extender, max_length_sentences):
    #==                       Variables                       ==#
    semantics  = []
    sentences  = []
    sents_lens = []
    sema_lens  = []

    typerror = 'Entered wrong type!'
    sentence_delim = '::snt'
    semantik_delim = '::smt'
    file_delim = '::file'

    max_length = sod(max_length_sentences, -1, isInt(max_length_sentences));
    inpath  = sod(inpath, typerror, isStr(inpath));
    outpath = sod(inpath+'.'+ output_extender , typerror, isStr(output_extender));

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
    StringToFile(outpath,
                 sentences,
                 semantics,
                 sentence_delim,
                 semantik_delim,
                 mw_value_sen,
                 mw_value_sem,
                 max_length)
    
    return None
