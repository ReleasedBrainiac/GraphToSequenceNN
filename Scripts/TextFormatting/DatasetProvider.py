# - *- coding: utf-8*-
'''
    Used Resources:
        => https://stackoverflow.com/questions/32382686/unicodeencodeerror-charmap-codec-cant-encode-character-u2010-character-m
        => https://www.pythonsheets.com/notes/python-rexp.html
'''

import re
from Scripts.TextFormatting.ContentSupport import isList, isStr, isInStr, isInt, isBool, isNone, isNotNone
from Scripts.TextFormatting.ContentSupport import setOrDefault
from Scripts.TreeHandler.TreeParser import Cleaner
from Scripts.GraphHandler.GraphTreeConverter import GetDataSet
from Scripts.AMRHandler.AMRCleaner import GenerateCleanAMR

'''
    This class library is used for content extraction from AMR dataset.
'''

TYP_ERROR_MESSAGE = 'Entered wrong type! Input is no String!'

def GetOutputPath(inpath, output_extender):
    """
    This function return a result output path depending on the given input path and a extender.
        :param inpath: raw data input path
        :param output_extender: result data path extender
    """
    if isStr(inpath) and isStr(output_extender):
        return setOrDefault(inpath+'.'+ output_extender , TYP_ERROR_MESSAGE, isStr(output_extender))
    else:
        print('WRONG INPUT FOR [GetOutputPath]')
        return None
    
def CalculateMeanValue(sentences_length):
    """
    This function calculate the mean over all values in a list.
        :param sentences_length: lengths of all sentences
    """
    if(isList(sentences_length)):
        sent_summ = 0

        for index, _ in enumerate(sentences_length):
            sent_summ += sentences_length[index]

        mw = int(round(sent_summ / len(sentences_length)))
        return mw
    else:
        print('WRONG INPUT FOR [CalculateMeanValue]')
        return None

def DatasetAsList(path):
    """
    This function provide a file reader for the AMR dataset.
        :param path: path string to dataset text file
    """
    if(isStr(path)):
        with open(path, 'r', encoding="utf8") as fileIn:
            data=fileIn.read()
            content=data.split('#')
            return content
    else:
        print('WRONG INPUT FOR [DatasetAsList]')
        return None

def SavingCorpus(sentence, semantic):
    """
    This function build a simple concatenation string containing a sentence and a semantic.
        :param sentence: cleaned sentence with sentences flag
        :param semantic: cleaned correspondign semantic for the sentence with semantic flag
    """
    if isStr(sentence) and isNotNone(semantic):
            return sentence + semantic
    else:
        print('WRONG INPUT FOR [SavingCorpus]')
        return None

def RestrictionCorpus(max_len, sentence, semantic):
    """
    This funtion check a sentence and semantic pair satisfy the size restictions.
        :param max_len: max allows length of a sentence
        :param sentence: the cleaned sentence
        :param semantic: the cleaned correspondign semantic for the sentence
    """
    if isInt(max_len) and isStr(sentence) and isNotNone(semantic):

        sen_size = max_len
        sem_size = max_len*2

        if (max_len < 1) or ((len(sentence) < (sen_size+1)) and (len(semantic) < (sem_size+1))):
            return [sentence, semantic]

    else:
        print('WRONG INPUT FOR [RestrictionCorpus]')
        return None

def CleanSentence(in_sentence):
    """
    This method clean up the a given amr extracted sentence from text formating markup.
        :param in_sentence: raw sentence AMR split element 
    """
    if(isStr(in_sentence)):
        in_sentence = re.sub('<[^/>][^>]*>','', in_sentence)
        in_sentence = re.sub('</[^>]+>','', in_sentence)
        in_sentence = re.sub('<[^/>]+/>','', '#'+in_sentence)
        return in_sentence+'\n'
    else:
        print('WRONG INPUT FOR [CleanSentence]')
        return None

def ReforgeSemanticToCleanMinimalARM(semantic, sem_flag):
    """
    This function allows to convert a raw AMR semantic (graph-) string  
    into a cleaned and minimalized version of it.
        :param semantic: raw semantic input
        :param sem_flag: marker/delim to add to cleaned semantic
    """
    if isStr(semantic) and isStr(sem_flag):
        store = semantic
        semantic = GenerateCleanAMR(semantic, '(', ')')

        if isNone(semantic):
            print(store)

        half_cleaned_sem = '#'+sem_flag+' \n'+semantic+'\n'
        out = half_cleaned_sem+'\n'
        return out
    else:
        print('WRONG INPUT FOR [ReforgeSemanticToCleanMinimalARM]')
        return None

def ReforgeSemanticToMinimalAnyTree(semantic, sem_flag, print_console, to_process):
    """
    This function allows to convert a raw AMR semantic (graph-) string
    into a cleaned and minimalized anytree reprenstation of it.
        :param semantic: raw semantic input
        :param sem_flag: marker/delim to add to cleaned semantic
        :param print_console: If True you get console output for GraphSalvage.Cleaner
        :param to_process: If False you get a JsonString for saving in file.
    """
    if(isStr(semantic)) and (isStr(sem_flag)) and (isBool(print_console)) and (isBool(to_process)):
        half_cleaned_sem = '#'+sem_flag+' '+semantic+'\n'
        out = half_cleaned_sem+'\n'
        return Cleaner(out, sem_flag, print_console, to_process)
    else:
        print('WRONG INPUT FOR [ReforgeSemanticToMinimalAnyTree]')
        return None

def GetSingleDatasetPair(sent_flag, sem_flag, sentence, semantic, want_as_arm, isShowConsole, isNotStoring):
    """
    This function collect a data pair from raw sentences and semantics.
    ATTENTION: [want_as_arm = true] does not support conversion with ConvertToTensorMatrices!
        :param sent_flag: a marker to attach to the cleaned sentence, make it easier to find later
        :param sem_flag: a marker to attach to the cleaned semantic, make it easier to find later
        :param sentence: raw input of the AMR sentence
        :param semantic: raw input of the AMR semantic
        :param want_as_arm: True then it return a tree-like formated AMR string for the semantic entry
        :param isShowConsole: True then it show in and out of the GraphSalvage.Cleaner on console
        :param isNotStoring: True you get GraphTree as AnyTree for further usage, else you get a JsonString for saving in file
    """
    if (isStr(sent_flag)) and (isStr(sem_flag)) and (isStr(sentence)) and (isStr(semantic)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
        sentence = sent_flag+' '+sentence
        sentence = CleanSentence(sentence)
        if(want_as_arm):
            semantic = ReforgeSemanticToCleanMinimalARM(semantic, sem_flag)
            return [sentence, semantic]
        else:
            semantic = ReforgeSemanticToMinimalAnyTree(semantic, sem_flag, isShowConsole, isNotStoring)
            return [sentence, semantic]
    else:
        print('WRONG INPUT FOR [GetSingleDatasetPair]')
        return [None, None]

def GetMultiDatasetPairs(sent_flag, sem_flag, sent_array, sem_array, want_as_arm, isShowConsole, isNotStoring):
    """
    This function collect multiples pairs of semantic and sentence data as list of data pairs.
    For this case we pass arrays of raw sentences and semantics, 
    where index i in both arrays point to a sentence and the corresponding semantic.
        :param sent_flag: a marker to attach to the cleaned sentence, make it easier to find later
        :param sem_flag: a marker to attach to the cleaned semantic, make it easier to find later
        :param sent_array: Array of the raw input of the AMR sentences
        :param sem_array: Array of the raw input of the AMR semantics
        :param want_as_arm: True then it return a tree-like formated AMR string for the semantic entry
        :param isShowConsole: True then it show in and out of the GraphSalvage.Cleaner on console
        :param isNotStoring: True you get GraphTree as AnyTree for further usage, else you get a JsonString for saving in file
    """
    if (isStr(sent_flag)) and (isStr(sem_flag)) and (isList(sent_array)) and (isList(sem_array)) and (isBool(want_as_arm)) and (isBool(isShowConsole)) and (isBool(isNotStoring)):
        dataset_pairs_sent_sem = []
        for i in range(min(len(sent_array), len(sem_array))):
            dataset_pairs_sent_sem.append(GetSingleDatasetPair(sent_flag, sem_flag, sent_array[i], sem_array[i], want_as_arm, isShowConsole, isNotStoring))

        return dataset_pairs_sent_sem
    else:
        print('WRONG INPUT FOR [GetMultiDatasetPair]')
        return []

def ExtractSentence(x_delim, in_content, index):
    """
    This function extract the sentence element from AMR corpus element!
        :param x_delim: marker/delim to find the sentence fragment
        :param in_content: raw amr string fragment from split of full AMR dataset string
        :param index: position where the sentence was found
    """
    raw_start_index = in_content[index].find(x_delim)+6
    sentence = in_content[index]
    sent_len = len(sentence)
    return sentence[raw_start_index:sent_len-1]

def ExtractSemantic(in_content, index):
    """
    This function extract the semantics element from AMR corpus element!
        :param in_content: raw amr string fragment from split of full AMR dataset string
        :param index: position where the semantic was found
    """
    raw_content = in_content[index].split('.txt')
    raw_con_index = len(raw_content)-1
    return raw_content[raw_con_index]

def ExtractContent(in_content, max_len, x_delim, y_delim):
    """
    This function collect the AMR-String-Representation and the corresponding sentence from AMR corpus.
        :param in_content: raw amr string fragment from split of full AMR dataset string
        :param max_len: maximal allowed length of a sentence and semantics
        :param x_delim: marker/delim to validate fragment as raw sentence
        :param y_delim: marker/delim to validate fragment as raw semantic
    """
    if isNotNone(in_content) and isStr(x_delim) and isStr(y_delim):
        sentence = ''
        semantic = ''
        result_pair = None
        sentence_found = False
        semantic_found = False
        sent_lens = []
        sem_lens = []
        sentences = []
        semantics = []

        for index, elem in enumerate(in_content):
            if (x_delim in elem):
                sentence = ExtractSentence(x_delim, in_content, index)
                sentence_found = True

            if (y_delim in elem) and (not semantic_found):
                semantic = ExtractSemantic(in_content, index)
                semantic_found = True

            if sentence_found and semantic_found:
                result_pair = RestrictionCorpus(max_len, sentence, semantic)
                sentence_found = False
                semantic_found = False

            if isNotNone(result_pair):
                sent_lens.append(len(result_pair[0]))
                sem_lens.append(len(result_pair[1]))
                sentences.append(result_pair[0])
                semantics.append(result_pair[1])

        if(len(sent_lens) == len(sem_lens) == len(sentences) == len(semantics)):
            return [sent_lens, sem_lens, sentences, semantics]
        else:
            print('WRONG OUTPUT FOR [ExtractContent]... Size of outputs dont match!')
            return None
    else:
        print('WRONG INPUT FOR [ExtractContent]')
        return None
   
def SaveToFile(path, len_sen_mw, len_sem_mw, max_len, data_pairs):
    """
    This function save the collected content to a given file.
        :param path: path to output file 
        :param len_sen_mw: mean of sentences length
        :param len_sem_mw: mean of semantics length
        :param max_len: max length of sentences we desire to store
        :param data_pairs: result data pairs as list
    """
    with open(path, 'w', encoding="utf8") as fileOut:
        for i in range(len(data_pairs)):
            result = SavingCorpus(data_pairs[i][0], data_pairs[i][1])
            if isNotNone(result):
                fileOut.write(result)
                fileOut.flush()

        print(path)
        return None

def BasicPipeline(inpath, output_extender, max_length, save_as_arm, print_console, is_not_saving):
    """
    This function collect the cleaned sentence graphs as:
    1. AMR string representation [save_as_arm = True]
    2. AnyTree [else]:
        1. as JSON     [is_not_saving = true]
        2. as AnyNode  [else]

        :param inpath: path of dataset text file
        :param output_extender: extender to define result filename
        :param max_length: max allows length for sentences
        :param save_as_arm: output will be save as tree like formated AMR string
        :param print_console: show all reforging at the Cleaner on console
        :param is_not_saving: set result to JSON or AnyNode [save_as_arm = False]
    """
    # Carrier Arrays/Lists
    semantics  = []
    sentences  = []
    sents_lens = []
    sema_lens  = []

    # AMR preprocessing Constants
    TYP_ERROR = 'Entered wrong type! Input is no String!'
    SENTENCE_DELIM = '::snt'
    SEMANTIC_DELIM = '::smt'
    FILE_DELIM = '::file'

    max_length = setOrDefault(max_length, -1, isInt(max_length))
    inpath  = setOrDefault(inpath, TYP_ERROR, isStr(inpath))

    dataset = DatasetAsList(inpath)

    len_dataset = len(dataset)
    dataset=dataset[1:len_dataset]
    sents_lens, sema_lens, sentences, semantics = ExtractContent(dataset, max_length,SENTENCE_DELIM, FILE_DELIM)

    mw_value_sen = CalculateMeanValue(sents_lens)
    mw_value_sem = CalculateMeanValue(sema_lens)

    data_pairs = GetMultiDatasetPairs(SENTENCE_DELIM, SEMANTIC_DELIM, sentences, semantics, save_as_arm, print_console, is_not_saving)

    print('Max restriction: ', max_length)
    print('Path input: ', inpath)
    print('Count sentences: ', len(sentences))
    print('Count semantics: ', len(semantics))
    print('Mean sentences: ', mw_value_sen)
    print('Mean semantics: ', mw_value_sem)

    return [mw_value_sen, mw_value_sem, max_length, data_pairs]

def SavePipeline(inpath, output_extender, max_length, save_as_arm, print_console):
    """
    This function calls the BasicPipeline and store the desired results.
        :param inpath: path of dataset text file
        :param output_extender: extender to define result filename
        :param max_length: max allows length for sentences
        :param save_as_arm: output will be save as tree like formated AMR string
        :param print_console: show all reforging at the Cleaner on console
    """
    mw_value_sen, mw_value_sem, max_length, data_pairs = BasicPipeline(inpath, output_extender, max_length, save_as_arm, print_console, False)
    outpath = GetOutputPath(inpath, output_extender)

    if (print_console):
        print('outpath: ', outpath)

    SaveToFile(outpath,
               mw_value_sen,
               mw_value_sem,
               max_length,
               data_pairs)
    
    return None

def DataPipeline(inpath, output_extender, max_length, save_as_arm, print_console):
    """
    This function calls the BasicPipeline and return the cleaned dataset for ANN usage.
        :param inpath: path of dataset text file
        :param output_extender: extender to define result filename
        :param max_length: max allows length for sentences
        :param save_as_arm: output will be save as tree like formated AMR string
        :param print_console: show all reforging at the Cleaner on console
    """
    if(save_as_arm == True):
        print('Processing dataset on AMR string representation not supported! Please set [save_as_arm=FALSE]!')
        return None
    else:
        data_pairs = BasicPipeline(inpath, output_extender, max_length, save_as_arm, print_console, True)[3]
        return GetDataSet(data_pairs)