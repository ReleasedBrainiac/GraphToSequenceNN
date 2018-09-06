# - *- coding: utf-8*-
import re
from TextFormatting.ContentSupport import isInStr, isNotInStr, isNotNone, isStr, isInt, toInt, isNumber
from TextFormatting.ContentSupport import hasContent, GetRandomInt

# Gathered definitions of AMR semantic strings from AMR Datasets
INDENTATION = 6
COLON = ':'
QUOTATION_MARK = '"'
NEG_POLARITY = 'not'
NEG_POL_LABEL = 'NT0'

NEW_NODES_DICT = {}

EXTENSIO_REGEX = '\-\d+'
POLARITY_SIGN_REGEX = '\s+\-\s*'
QUALIFIED_STR_REGEX = '\B(\"\w+( \w+)*\")'
FLAG_REGEX = '\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'
ARGS_REGEX = '\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'

def CreateNewLabel(number):
    """
    This function create a label with a upper-case post- and pre-element.
    The passed number (maybe an iteration value) is placed between them.
        :param number: a number which is placed in the label
    """
    inner_index = GetRandomInt(0, 50)
    if isNumber(number) and (number > -1):
        inner_index = number
        
    return 'Y' + str(inner_index) + 'Z'

def CreateNewDefinedNode(label , content, open_par, close_par):
    """
    This function defines a new AMR converted node element.
    Depending on the content it returns a node with label and content or just containing a label.
        :param label: the desired node label
        :param content: the corresponding content
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(label) and isStr(open_par) and isStr(close_par):
        if isStr(content):
            return open_par + label + ' / ' + content + close_par
        else:
            return open_par + label + close_par
    else:
        print('WRONG INPUT FOR [CreateNewDefinedNode]')
        return None

def CountLeadingWhiteSpaces(raw_line):
    """
    This function count leading whitespaces in raw line of a raw or indentation cleaned AMR string.
    Remind the input should not be preformated (e.g. cleaning from whitespaces).
        :param raw_line: a raw or indentation cleaned AMR string.
    """
    if isStr(raw_line):
        if isInStr(' ', raw_line):
            return (len(raw_line) - len(raw_line.lstrip(' ')))
        else:
            return 0
    else:
        print('WRONG INPUT FOR [CountLeadingWhiteSpaces]')
        return None

def GetCurrentDepth(raw_line):
    """
    This function calculate the depth of a AMR strig node element depending on:
    1. the amount of leading whitespaces
    2. the global defined INDENTATION constant
        :param raw_line: a raw or indentation cleaned AMR string.
    """
    if isStr(raw_line):
        if isInStr('', raw_line):
            return toInt(CountLeadingWhiteSpaces(raw_line) / INDENTATION)
        else:
            return 0
    else:
        print('WRONG INPUT FOR [GetCurrentDepth]')
        return None

def CountSubsStrInStr(content_str, search_element):
    """
    This function count occourences of srtings inside another string.
        :param content_str: the string were want to discover occourences
        :param search_element: the string we are searching for
    """
    if isStr(content_str) and isStr(search_element) and (len(content_str) > len(search_element)):
        return content_str.count(search_element)
    else: 
        print('WRONG INPUT FOR [CountSubsStrInStr]')
        return 0

def CheckOpenEnclosing(content, open_par, close_par):
    """
    This function check the equal amount of opened and closed parenthesis depending on the given parenthesis definition.
        :param content: raw string containing desired parenthesis
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(content) and isStr(open_par) and isStr(close_par) and isInStr(open_par, content) and isInStr(close_par, content) :
        count_open = CountSubsStrInStr(content,open_par)
        count_close = CountSubsStrInStr(content,close_par)

        if (count_open == count_close):
            return True
        else:
            return False
    else:
        print(isInStr(open_par, content))
        print(isInStr(close_par, content))
        print('WRONG INPUT FOR [CheckOpenEnclosing]')
        return None

def GetEnclosedContent(content, open_par, close_par):
    """
    This function return the most outer nested content in a string by given desired parenthesis strings.
        :param content: string containing content nested in desired parenthesis.
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if  isStr(content) and isStr(open_par) and isStr(close_par):
        if isInStr(open_par, content) and isInStr(close_par, content):
            pos_open = content.index(open_par)
            pos_close = content.rfind(close_par)
            return content[pos_open+1:pos_close]
        else:
            return content
    else:
        print('test', content)
        print('WRONG INPUT FOR [GetEnclosedContent]')
        return None

def EncloseSoloLabels(raw_line):
    """
    This function create new AMR nodes on argument flags following unenclosed labels.
        :param raw_line: a string containing argument flags following unenclosed labels
    """
    if isStr(raw_line):
        if isInStr(COLON, raw_line):
            loot = re.findall(ARGS_REGEX, raw_line)

            for loot_elem in loot:
                joined_elem_regex = ''.join(loot_elem)
                joined_elem_replace = ''.join([loot_elem[0], '('+loot_elem[1].lstrip(' ')+')'])
                raw_line = re.sub(joined_elem_regex, joined_elem_replace, raw_line)      
    
        return raw_line
    else:
        print('WRONG INPUT FOR [EncloseSoloLabels]')
        return None

def EncloseQualifiedStringInforamtions(raw_line, open_par, close_par):
    """
    This function creates new AMR nodes with desired parenthesis for Qualified Names enclosed in quotation marks.
        :param raw_line: a string with (at least one) qualified name(s)
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(raw_line):
        if isInStr(QUOTATION_MARK, raw_line):
            loot = re.findall(QUALIFIED_STR_REGEX, raw_line)
            run_iter = -1

            for loot_elem in loot:
                found_elem = loot_elem[0]
                label = None
                content = None

                if isNotNone(found_elem) and hasContent(found_elem) and found_elem.count(QUOTATION_MARK) == 2:
                    if found_elem in NEW_NODES_DICT:
                        label = NEW_NODES_DICT[found_elem]
                        content = None

                    else:     
                        run_iter = run_iter + 1
                        label = CreateNewLabel(run_iter)
                        content = re.sub(QUOTATION_MARK,'',found_elem)
                        NEW_NODES_DICT[found_elem] = label
                    
                    replace = CreateNewDefinedNode(label, content, open_par, close_par)
                    raw_line = raw_line.replace(found_elem, replace, 1)

                else:
                    print('WRONG DEFINITION ['+ found_elem + ']FOUND IN INPUT FOR [EncloseQualifiedStringInforamtions]')
                    continue   

        return raw_line
    else:
        print('WRONG INPUT FOR [EncloseQualifiedStringInforamtions]')
        return None

def ReplacePolarity(raw_line, open_par, close_par):
    """
    This function replace negative polarity (-) signs in a raw semantic line with a new AMR node.
        :param raw_line: string containing (at least on) AMR polarity sign
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(raw_line):
        if isInStr(' - ', raw_line):
            next_depth = GetCurrentDepth(raw_line) + 1 
            label = None
            content = None

            if NEG_POLARITY in NEW_NODES_DICT:
                label = NEW_NODES_DICT[NEG_POLARITY]
                content = None
            else:
                label = NEG_POL_LABEL
                content = NEG_POLARITY
                NEW_NODES_DICT[NEG_POLARITY] = NEG_POL_LABEL

            replace_node_str = CreateNewDefinedNode(label, content, open_par, close_par)
            replace = AddLeadingSpace(replace_node_str, next_depth)
            return re.sub(POLARITY_SIGN_REGEX, ('\n'+ replace), raw_line)
        else:
            return raw_line   
    else:
        print('WRONG INPUT FOR [ReplacePolarity]')
        return None

def DeleteFlags(raw_line):
    """
    This function delete AMR flags and only keep the informations they were flagged.
        :param raw_line: string with (at least on) AMR flag(s)
    """
    if isStr(raw_line):
        if isInStr(COLON, raw_line):
            return re.sub(FLAG_REGEX, '', raw_line)            
        else:
            return raw_line
    else:
        print('WRONG INPUT FOR [DeleteFlags]')
        return None

def DeleteWordExtension(raw_line):
    """
    This function delete word extensions from node content in a AMR semantic line fragment.
        :param raw_line: a AMR semantic line fragment
    """
    if isStr(raw_line):
        if isInStr('-', raw_line):
            return re.sub(EXTENSIO_REGEX,'', raw_line)
        else:
            return raw_line
    else:
        print('WRONG INPUT FOR [DeleteWordExtension]')
        return None
    
def ExploreAdditionalcontent(raw_line, open_par, close_par):
    """
    This function search in a AMR line fragment about additional context for the AMR node.
        :param raw_line: a AMR line fragment with a node and maybe additional context 
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(raw_line):
        result = raw_line

        if isInStr(COLON, result):
            result = DeleteFlags(result)
        if isInStr('-', result):
            result = ReplacePolarity(result, open_par, close_par)
            result = DeleteWordExtension(result)
                
        return result
    else:
        print('WRONG INPUT FOR [ExploreAdditionalcontent]')
        return None

def GetUnformatedAMRString(raw_amr):
    """
    This function replace all line and space formatting in raw AMR (graph-) string with a sigle whitespace.
        :param raw_amr: a raw AMR (graph-) string from AMR Dataset
    """
    if isStr(raw_amr):
        return (' '.join(raw_amr.split()))
    else:
        print('WRONG INPUT FOR [GetUnformatedAMRString]')
        return None

def AddLeadingSpace(str, depth):
    """
    This function add a desired depth to a cleaned AMR line fragment by means of leading whitespaces.
    This will keep the AMR datasets indentation definition of a AMR semantice (graph) string.
        :param str: the cleaned AMR semantic line fragment
        :param depth: the desired depth rather intended position in the string
    """
    if(isStr(str)) and (isInt(depth)):
        for _ in range((depth * INDENTATION)):
            str = ' '+str
        return str

    else:
        print('WRONG INPUT FOR [AddLeadingSpace]')
        return None

def NiceFormatting(amr_str, open_par, close_par):
    """
    This function format and clean up a raw AMR semantic (graph-) string.
    This allow to clean up format definitions and remove [for this project] uninteresting content!
        :param amr_str: a raw AMR semantic (graph-) string
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(amr_str) and isStr(open_par) and isStr(close_par):
        depth = -1
        openings = amr_str.split(open_par)
        struct_contain = []

        for line in openings:
            depth = depth + 1
            new_line = AddLeadingSpace((open_par + line), depth)

            if isInStr(close_par, new_line):
                occourences = CountSubsStrInStr(new_line, close_par)
                depth = depth - occourences
            
            if isInStr(COLON, new_line):
                new_line = ExploreAdditionalcontent(new_line, open_par, close_par)
            
            struct_contain.append(new_line)

        returning = '\n'.join(struct_contain) + ')'
        return returning

    else:
        print('WRONG INPUT FOR [NiceFormatting]')
        return None

def GenerateCleanAMR(raw_amr, open_par, close_par):
    """
    This function preprocess a raw AMR semantic (graph-) string for further usage in the main project.
        :param raw_amr: a raw AMR semantic (graph-) string
        :param open_par: string of desired parenthesis type style "open"
        :param close_par: string of desired parenthesis type style "close"
    """
    if isStr(raw_amr) and isStr(open_par) and isStr(close_par):
        unformated_str = GetUnformatedAMRString(raw_amr)
        if CheckOpenEnclosing(unformated_str, open_par, close_par):
            node_enclosed_str = EncloseSoloLabels(unformated_str)
            name_enclosed_str = EncloseQualifiedStringInforamtions(node_enclosed_str, open_par, close_par)
            amr_str = GetEnclosedContent(name_enclosed_str, open_par, close_par)

            print('unformated_str:', unformated_str)
            print('node_enclosed_str:',node_enclosed_str)
            print('name_enclosed_str:',name_enclosed_str)
            print('amr_str:',amr_str)
            return NiceFormatting(amr_str, open_par, close_par) 
        else:
            print('UNEQUAL AMOUNT OF BRACKET PAIRS IN INPUT FOR [GenerateCleanAMR]')
            return None
    else:
        print('WRONG INPUT FOR [GenerateCleanAMR]')
        return None
