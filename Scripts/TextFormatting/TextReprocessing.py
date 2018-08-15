#This script was created by T.Turke for content extraction from AMR dataset.
#4 use on windows cmd ~> https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
#~> https://www.datacamp.com/community/tutorials/reading-writing-files-python
#cmd write => chcp 65001

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

#==             Record clean_content extractor            ==#
def ExtractContent(in_content, x_delim, y_delim):
    sent_lens = []
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
            raw_content = in_content[index]
            raw_con_index = in_content[index].find('(')
            sem_len = len(raw_content)
            semantic = raw_content[raw_con_index:sem_len-1]
            semantics.append(semantic)

    return [sent_lens, sentences, semantics]
        

#===========================================================#
#                          Pipeline                         #
#===========================================================#

#==                       Variables                       ==#
inpath = '../../Datasets/Cleaned/amr-bank-struct-v1.6-dev.txt-cleaned-content.txt'
#inpath = 'sample of -MA dataset.txt'
sentence_delim = '::snt'
semantik_delim = '::smt'
semantics = []
sentences = []
sents_lens = []

#==                     Read Dataset                      ==#
result = FileToString(inpath)

#==             Collect relevant raw_content              ==#
len_dataset = len(result)
result=result[1:len_dataset]
sents_lens, sentences, semantics = ExtractContent(result, sentence_delim, semantik_delim)

#==                     Check Median                      ==#
mw_value = CalcMW(sents_lens)
print(mw_value)
print(len(sents_lens))






