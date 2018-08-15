#This script was created by T.Turke for content extraction from AMR dataset.
#4 use on windows cmd ~> https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
#~> https://www.datacamp.com/community/tutorials/reading-writing-files-python
#~> https://www.pythonsheets.com/notes/python-rexp.html
#cmd write => chcp 65001

import re

#===========================================================#
#                   Methods/Functions                       #
#===========================================================#

#==         Calculate MW for a list of numbers            ==#
def CalcMW(sentence_length):
    sent_summ = 0
    
    for index, elem in enumerate(sentence_length):
        sent_summ += sentence_length[index]
        
    mw = int(round(sent_summ / len(sentence_length)))
    return mw

#==                    Read AMR Dataset                   ==#
def FileToString(xpath):
    with open(xpath, 'r', encoding="utf8") as fileIn: 
        data=fileIn.read()
        result=data.split('#')
        return result

#==                     Filter Content                    ==#
def ClearSentence(in_sentence)
    in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
    in_sentence = re.sub('</[^>]+>','', in_sentence)
    in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
    in_sentence = re.sub()+'\n'
    return in_sentence
        

#==                    Write AMR Dataset                  ==#
def StringToFile(ypath, sent_array, sem_array, sent_flag, sem_flag, len_sen_mw, len_sem_mw, max_len):
    with open(ypath, 'w', encoding="utf8") as fileOut:
        allowed_size_sen = min(len_sen_mw, max_len)
        allowed_size_sem = min(len_sem_mw, (max_len*2))
        print(allowed_size_sen)
        print(allowed_size_sem)
        for i in range(min(len(sent_array), len(sem_array))):
            #Gather sentences without linking substrings because they are not contained in the semantic graph
            sent = sent_flag+' '+sent_array[i]
            sent = re.sub('<[^/>][^>]*>','', sent)
            sent = re.sub('</[^>]+>','', sent)
            sent = re.sub('<[^/>]+/>','', '#'+sent)+'\n'
            #sent = '#'+sent_flag+' '+sent_array[i]+'\n'
            #Gather semantic graphs as string lines because the '\n and '\t' are the start and end symbols the network
            half_cleaned_sem = '#'+sem_flag+' '+sem_array[i].replace('\n', '')+'\n'
            sem = re.sub(' +',' ',half_cleaned_sem)+'\n'
            #sem = '#'+sem_flag+' '+sem_array[i]+'\n'

            if (max_len < 1):
                out = sent + sem
                fileOut.write(out)
                fileOut.flush()
            
            if (max_len > 0) and (len(sent) < (allowed_size_sen+1)) and (len(sem) < (allowed_size_sem+1)):
                out = sent + sem
                fileOut.write(out)
                fileOut.flush()
    return None

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

#==                       Variables                       ==#
max_length = -1
semantics = []
sentences = []
sents_lens = []
sema_lens = []

inpath = '../../Datasets/Raw/Der Kleine Prinz AMR/amr-bank-struct-v1.6-dev.txt'
#inpath = 'sample of -MA dataset.txt'
outpath = inpath+'-'+'cleaned-content.txt'
sentence_delim = '::snt'
semantik_delim = '::smt'
file_delim = '::file'

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



