# - *- coding: utf-8*-
from TextFormatting.ContentSupport import isInStr, isNotInStr, isNotNone, isStr, isInt

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
    if  isStr(content) and isStr(open_par) and isStr(closing_par) and isInStr(open_par, content) and isInStr(closing_par, content):

        pos_open = content.index(open_par)
        pos_close = content.rfind(closing_par)
        return content[pos_open+1:pos_close]
    else:
        print('WRONG INPUT FOR [GetEnclosedContent]')
        return None


def ExploreAdditionalcontent(substring):
    if isStr(substring):
        print()
    else:
        print('WRONG INPUT FOR [ExploreAdditionalcontent]')
        return None

def GetUnformatedAMRString(raw_amr):
    if isNotNone(raw_amr) and isStr(raw_amr):
        return ' '.join(raw_amr.split())
    else:
        print('WRONG INPUT FOR [GetUnformatedAMRString]')
        return None

def AddLeadingSpace(str, amount):
    if(isStr(str)) and (isInt(amount)):
        for _ in range(amount):
            str = '      '+str
        return str

    else:
        print('WRONG INPUT FOR [AddLeadingSpace]')
        return None

def RuleFormatting(amr_str, open_par, closing_par):
    if isNotNone(amr_str) and isStr(amr_str):
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
                start_occourence = new_line.find(':')
                end_line = len(new_line)
                sub_line = new_line[start_occourence:end_line]
                print(sub_line, 'contain whites => ', isInStr(' ', sub_line))
            
            #print(new_line)
            struct_contain.append(new_line)

        return None

    else:
        print('WRONG INPUT FOR [RuleFormatting]')
        return None


def BuildCleanDefinedAMR(raw_amr, open_par, closing_par):
    if isNotNone(raw_amr) and isStr(raw_amr):
        amr_str = GetEnclosedContent(GetUnformatedAMRString(raw_amr), open_par, closing_par)
        new_amr = RuleFormatting(amr_str, open_par, closing_par)
        #return (open_par + new_amr + closing_par)
    else:
        print('WRONG INPUT FOR [BuildCleanDefinedAMR]')
        return None
