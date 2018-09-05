# - *- coding: utf-8*-
import re
from TextFormatting.ContentSupport import isInStr, isNotInStr, isNotNone, isStr, isInt, toInt
INDENTATION = 6

def CountLeadingWhiteSpaces(raw_line):
    if isStr(raw_line):
        if isInStr(' ', raw_line):
            return (len(raw_line) - len(raw_line.lstrip(' ')))
        else:
            return 0
    else:
        print('WRONG INPUT FOR [CountLeadingWhiteSpaces]')
        return None

def GetCurrentDepth(raw_line):
    if isStr(raw_line):
        if isInStr('', raw_line):
            return toInt(CountLeadingWhiteSpaces(raw_line) / INDENTATION)
        else:
            return 0
    else:
        print('WRONG INPUT FOR [GetCurrentDepth]')
        return None

def CreateEncapsulatedDepthRefinedReplacement(raw_origin_line, replacement):
    if isStr(raw_origin_line):
        depth = GetCurrentDepth(raw_origin_line)
        return ('\n'+AddLeadingSpace(replacement.lstrip(' '), (depth+1)))
    else:
        print('WRONG INPUT FOR [ReplacePolarity]')
        return None

def CountSubsStrInStr(content_str, search_element):
    if isStr(content_str) and isStr(search_element):
        return content_str.count(search_element)
    else: 
        print('WRONG INPUT FOR [CountSubsStrInStr]')
        return 0

def CheckOpenEnclosing(content, open_par, closing_par):
    if isStr(content) and isStr(open_par) and isStr(closing_par):
        count_open = CountSubsStrInStr(content,open_par)
        count_close = CountSubsStrInStr(content,closing_par)

        if (count_open == count_close):
            return True
        else:
            return False
    else:
        print('WRONG INPUT FOR [CheckOpenEnclosing]')
        return None

def GetEnclosedContent(content, open_par, closing_par):
    if  isStr(content) and isStr(open_par) and isStr(closing_par):
        if isInStr(open_par, content) and isInStr(closing_par, content):
            pos_open = content.index(open_par)
            pos_close = content.rfind(closing_par)
            return content[pos_open+1:pos_close]
        else:
            return content
    else:
        print('WRONG INPUT FOR [GetEnclosedContent]')
        return None

def EncloseSoloLabels(raw_line):
    if isStr(raw_line):
        if isInStr(':', raw_line):
            ARGS_REGEX = '\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'
            loot = re.findall(ARGS_REGEX, raw_line)

            for loot_elem in loot:
                joined_elem_regex = ''.join(loot_elem)
                joined_elem_replace = ''.join([loot_elem[0], '('+loot_elem[1].lstrip(' ')+')'])
                raw_line = re.sub(joined_elem_regex, joined_elem_replace, raw_line)      

            return raw_line
    else:
        print('WRONG INPUT FOR [DeleteFlags]')
        return None

def DeleteFlags(raw_line):
    if isStr(raw_line):
        if isInStr(':', raw_line):

            FLAG_REGEX = '\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'
            return re.sub(FLAG_REGEX, '', raw_line)            
        else:
            return raw_line
    else:
        print('WRONG INPUT FOR [DeleteFlags]')
        return None

def ReplacePolarity(raw_line):
    if isStr(raw_line):
        if isInStr('-', raw_line):
            POLARITY_SIGN_REGEX = '\s+\-\s*'
            depth = GetCurrentDepth(raw_line)
            replacement = AddLeadingSpace('(not)', (depth+1))
            return re.sub(POLARITY_SIGN_REGEX, ('\n'+ replacement), raw_line)
        else:
            return raw_line   
    else:
        print('WRONG INPUT FOR [ReplacePolarity]')
        return None

def DeleteWordExtension(raw_line):
    if isStr(raw_line):
        if isInStr('-', raw_line):
            EXTENSIO_REGEX = '\-\d+'
            return re.sub(EXTENSIO_REGEX,'', raw_line)
        else:
            return raw_line
    else:
        print('WRONG INPUT FOR [DeleteWordExtension]')
        return None
    
def ExploreAdditionalcontent(string_raw):
    if isStr(string_raw):
        result = string_raw

        if isInStr(':', result):
            result = DeleteFlags(result)
        if isInStr('-', result):
            result = ReplacePolarity(result)
            result = DeleteWordExtension(result)
                
        return result
    else:
        print('WRONG INPUT FOR [ExploreAdditionalcontent]')
        return None

def GetUnformatedAMRString(raw_amr):
    if isStr(raw_amr):
        return ' '.join(raw_amr.split())
    else:
        print('WRONG INPUT FOR [GetUnformatedAMRString]')
        return None

def AddLeadingSpace(str, amount):
    if(isStr(str)) and (isInt(amount)):
        for _ in range((amount * INDENTATION)):
            str = ' '+str
        return str

    else:
        print('WRONG INPUT FOR [AddLeadingSpace]')
        return None

def NiceFormatting(amr_str, open_par, closing_par):
    if isStr(amr_str) and isStr(open_par) and isStr(closing_par):
        depth = -1
        openings = amr_str.split(open_par)
        struct_contain = []

        for line in openings:
            depth = depth + 1
            new_line = AddLeadingSpace((open_par + line), depth)

            if isInStr(closing_par, new_line):
                occourences = CountSubsStrInStr(new_line, closing_par)
                depth = depth - occourences
            
            if isInStr(':', new_line):
                new_line = ExploreAdditionalcontent(new_line)
            
            struct_contain.append(new_line)

        returning = '\n'.join(struct_contain)
        return returning

    else:
        print('WRONG INPUT FOR [NiceFormatting]')
        return None

def GenerateCleanAMR(raw_amr, open_par, closing_par):
    if isStr(raw_amr) and isStr(open_par) and isStr(closing_par):
        unformated_str = GetUnformatedAMRString(raw_amr)
        node_enclosed_str = EncloseSoloLabels(unformated_str)
        amr_str = GetEnclosedContent(node_enclosed_str, open_par, closing_par)
        return NiceFormatting(amr_str, open_par, closing_par) 
    else:
        print('WRONG INPUT FOR [GenerateCleanAMR]')
        return None
