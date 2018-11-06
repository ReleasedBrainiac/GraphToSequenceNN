# - *- coding: utf-8*-
import re
from DatasetHandler.ContentSupport import isInStr, isNotInStr, isNotNone, isStr, isInt, toInt, isNumber, isDict, isBool
from DatasetHandler.ContentSupport import hasContent, GetRandomInt
from Configurable.ProjectConstants import Constants

class Cleaner:
    # Variables inits
    node_parenthesis = ['(',')']
    edge_parenthesis = ['[',']']
    extension_dict = {}
    extension_keys_dict = {}
    new_nodes_dict = {}
    context = None
    cleaned = None
    isCleaned = False
    gotContext = False
    gotExtentsionsDict = False
    keep_edge_encoding = False

    # Class inits
    constants = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, open_bracket='(', close_bracket=')', input_context=None, input_extension_dict={}, keep_edges=False):
        try:

            self.constants = Constants()
            self.new_nodes_dict = {}

            if isStr(input_context): 
                self.gotContext = True
                self.context = input_context

            if isDict(input_extension_dict):
                self.gotExtentsionsDict = True
                self.extension_dict = input_extension_dict
                self.extension_keys_dict = self.extension_dict.keys()

            if isStr(open_bracket) and isStr(close_bracket):
                self.node_parenthesis[0] = open_bracket
                self.node_parenthesis[1] = close_bracket

            if isBool(keep_edges):
                self.keep_edge_encoding = keep_edges

            if(self.gotContext):
                self.GenerateCleanAMR()
        except ValueError:
            print("No valid context passed.")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def HasColon(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.COLON in in_context
        elif self.gotContext:
            return self.constants.COLON in self.context
        else:
            return False

    def HasQuotation(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.QUOTATION_MARK in in_context
        elif self.gotContext:
            return self.constants.QUOTATION_MARK in self.context
        else:
            return False

    def HasConnector(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.CONNECTOR in in_context
        elif self.gotContext:
            return self.constants.CONNECTOR in self.context
        else:
            return False
    
    def HasParenthesis(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.node_parenthesis[0] in in_context and self.node_parenthesis[1] in in_context
        elif self.gotContext:
            return self.node_parenthesis[0] in self.context and self.node_parenthesis[1] in self.context
        else:
            return False

    def HasPolarity(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return (self.constants.POLARITY in in_context or '(-)' in in_context)
        elif self.gotContext:
            return self.constants.POLARITY in self.context
        else:
            return False

    def HasWhitspaces(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.WHITESPACE in in_context
        elif self.gotContext:
            return self.constants.WHITESPACE in self.context
        else:
            return False

    def MatchSignsOccurences(self, in_context='', in_signs=['(',')']):
        if isStr(in_context) and len(in_context) > 0:
            count_sign_x = self.GetSignOccurenceCount(in_context, in_signs[0])
            count_sign_y = self.GetSignOccurenceCount(in_context, in_signs[1])
            return count_sign_x == count_sign_y
        elif self.gotContext:
            count_sign_x = self.GetSignOccurenceCount(self.context, in_signs[0])
            count_sign_y = self.GetSignOccurenceCount(self.context, in_signs[1])
            return count_sign_x == count_sign_y
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def CreateNewLabel(self, in_name=None, in_number=None):
        """
        This function create a label with a upper-case post- and pre-element.
        The passed number (maybe an iteration value) is placed between them.
            :param in_name: a desired label name
            :param in_number: a number which is placed in the label
        """
        try:
            name = 'NEW'
            index = GetRandomInt(0, 50)

            if isNumber(in_name) and (in_number > -1):
                index = in_number

            if isStr(in_name):
                name = in_name
                
            return str(name) + str(index)
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CreateNewLabel]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
 
    def CreateEdgeTagDefinition(self , in_content=None):
        """
        This function defines a new AMR converted node element.
        Depending on the content it returns a edge with content.
            :param in_content: the corresponding content
        """
        try:
            return self.node_parenthesis[0] + self.edge_parenthesis[0] + in_content + self.edge_parenthesis[1] + self.node_parenthesis[1]
        except ValueError:
            print("ERR: No label passed to [AMRCleaner.CreateEdgeTagDefinition].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CreateEdgeTagDefinition]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)        
        
    def CountLeadingWhiteSpaces(self, in_content):
        """
        This function count leading whitespaces in raw line of a raw or indentation cleaned AMR string.
        Remind the input should not be preformated (e.g. cleaning from whitespaces).
            :param in_content: a raw or indentation cleaned AMR string.
        """
        try:
            count = 0
            if self.HasWhitspaces(in_content):
                count = (len(in_content) - len(in_content.lstrip(' ')))

            return count
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.CountLeadingWhiteSpaces].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CountLeadingWhiteSpaces]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetSignOccurenceCount(self, in_content, in_search_element):
        """
        This function count occourences of srtings inside another string.
            :param in_content: the string were want to discover occourences
            :param in_search_element: the string we are searching for
        """
        try:
            occurence = 0
            if len(in_content) > len(in_search_element):
                occurence = in_content.count(in_search_element)

            return occurence
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.GetSignOccurenceCount].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.GetSignOccurenceCount]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetCurrentDepth(self, in_content):
        """
        This function calculate the depth of a AMR strig node element depending on:
        1. the amount of leading whitespaces
        2. the global defined constants.INDENTATION constant
            :param in_content: a raw or indentation cleaned AMR string.
        """
        try:
            depth = 0
            if self.HasWhitspaces(in_content):
                depth = toInt(self.CountLeadingWhiteSpaces(in_content) / self.constants.INDENTATION)

            return depth
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.GetCurrentDepth].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.GetCurrentDepth]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetEnclosedContent(self, in_content):
        """
        This function return the most outer nested content in a string by given desired node_parenthesis strings.
            :param in_content: string containing content nested in desired node_parenthesis.
        """
        try:
            if self.HasParenthesis(in_content):
                pos_open = in_content.index(self.node_parenthesis[0])
                pos_close = in_content.rfind(self.node_parenthesis[1])
                in_content = in_content[pos_open+1:pos_close]
            
            return in_content
        except ValueError:
            print("ERR: Missing or wrong value(s) passed to [AMRCleaner.GetEnclosedContent].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.GetEnclosedContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetUnformatedAMRString(self, in_content):
        """
        This function replace all line and space formatting in raw AMR (graph-) string with a sigle whitespace.
            :param in_content: a raw string from AMR Dataset
        """
        try:
            return (' '.join(in_content.split()))
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.GetUnformatedAMRString]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def CollectAllMatchesOfPattern(self, in_context, in_regex_pattern):
        try:
            return re.findall(in_regex_pattern, in_context)
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.CollectAllMatchesOfPattern].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CollectAllMatchesOfPattern]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceAllPatternMatches(self, in_pattern_search, in_replace, in_context):
        try:
            return re.sub(in_pattern_search, in_replace, in_context) 
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.ReplaceAllPatternMatches].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.ReplaceAllPatternMatches]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def AddLeadingSpace(self, str, depth):
        """
        This function add a desired depth to a cleaned AMR line fragment by means of leading whitespaces.
        This will keep the AMR datasets indentation definition of a AMR semantice (graph) string.
            :param str: the cleaned AMR semantic line fragment
            :param depth: the desired depth rather intended position in the string
        """
        try:
            for _ in range((depth * self.constants.INDENTATION)):
                str = ' '+str
            return str
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.AddLeadingSpace]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseEdge(self, in_match):
        """
            Encapsulate label inputs with a square brackets
            :param in_match: a label string component
        """   
        try:
            flag = in_match[0]
            flagged_element = '['+in_match[1].lstrip(' ')+']'
            return [flag, flagged_element]
        except ValueError:
            print("ERR: Missing or wrong value passed to [AMRCleaner.EncloseEdge].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.EncloseEdge]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseUnenclosedValues(self, in_content):
        """
        This function create new AMR nodes on argument flags following unenclosed labels.
            :param in_content: a string containing argument flags following unenclosed labels
        """
        try:
            if self.HasColon(in_content):
                for loot_elem in self.CollectAllMatchesOfPattern(in_content, self.constants.UNENCLOSED_ARGS_REGEX):
                    search = ''.join(loot_elem)
                    found_flag = loot_elem[0]
                    found_edge = loot_elem[1].lstrip(' ')
                    replace = ''.join([found_flag, ' ' + self.node_parenthesis[0] + found_edge + self.node_parenthesis[1]])

                    if 'ARG' not in loot_elem[0]:
                        if self.keep_edge_encoding:
                            found_flag, found_edge = self.EncloseEdge(loot_elem)
                            replace = ''.join([found_flag, ' ' + self.node_parenthesis[0] + found_edge + self.node_parenthesis[1]])
                        else:
                            replace = ''
                        
                    
                    in_content = self.ReplaceAllPatternMatches(search, replace, in_content)
            return in_content
        except ValueError:
            print("ERR: Missing or wrong value passed to [AMRCleaner.EncloseUnenclosedValues].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.EncloseUnenclosedValues]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseStringifiedValues(self, in_content):
        """
        This function creates new AMR nodes with desired node_parenthesis for Qualified Names enclosed in quotation marks.
            :param in_content: a string with (at least one) qualified name(s)
        """
        try:
            if self.HasQuotation(in_content):
                for elem in self.CollectAllMatchesOfPattern(in_content, self.constants.MARKER_NESTINGS_REGEX):
                    replacer = ''

                    if hasContent(elem[0]) and elem[0].count(self.constants.QUOTATION_MARK) == 2:
                        if self.keep_edge_encoding:
                            content = re.sub(self.constants.QUOTATION_MARK,'',elem[0]).replace('_',' ')

                            if all(x.isalnum() or x.isspace() for x in content):
                                replacer = self.CreateEdgeTagDefinition(content)

                    in_content = in_content.replace(elem[0], replacer, 1)

            return in_content
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.EncloseStringifiedValues].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.EncloseStringifiedValues]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def LookUpReplacement(self, in_content):
        try:
            if ('-' in in_content):
                look_up_control = self.CollectAllMatchesOfPattern(in_content, self.constants.EXTENSION_MULTI_WORD_REGEX)
                if (isNotNone(look_up_control) and isNotNone(self.extension_dict) and isDict(self.extension_dict)):
                    for found in look_up_control:
                        if len(found[0]) > 0 and found[0] in self.extension_keys_dict:
                            in_content = in_content.replace(found[0], self.extension_dict[found[0]])
                    
            return in_content
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.LookUpReplacement]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplacePolarity(self, in_content):
        """
        This function replace negative polarity (-) signs in a raw semantic line with a new AMR node.
            :param in_content: string containing (at least on) AMR polarity sign
        """
        try:
            content = self.constants.NEG_POLARITY
            replace = ''

            if self.keep_edge_encoding: replace = self.CreateEdgeTagDefinition(content)

            in_content = re.sub(self.constants.SIGN_POLARITY_REGEX, replace, in_content)
            return in_content
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.ReplacePolarity].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.ReplacePolarity]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteFlags(self, in_content):
        """
        This function delete AMR flags and only keep the informations they were flagged.
            :param in_content: string with (at least on) AMR flag(s)
        """
        try:
            if isInStr(self.constants.COLON, in_content):
                in_content = re.sub(self.constants.FLAG_REGEX, '', in_content)
            return in_content
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.DeleteFlags]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RemoveUnusedSignes(self, in_content):
        """
        This function delete all remaining signs we don't want to keep in the string.
            :param in_content: string
        """
        try:
            return re.sub(self.constants.SIGNS_REMOVE_UNUSED_REGEX, '', in_content)
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.RemoveUnusedSignes].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.RemoveUnusedSignes]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteWordExtension(self, in_content):
        """
        This function delete word extensions from node content in a AMR semantic line fragment.
            :param in_content: a AMR semantic line fragment
        """
        try:
            if isInStr('-', in_content):
                in_content = re.sub(self.constants.EXTENSION_NUMBER_REGEX,'', in_content)

            return in_content
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.DeleteWordExtension].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.DeleteWordExtension]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    '''  
    def QuantMapping(self, in_content):
        """
        This function replace concrete values with grouping string, which explain there qauntity more generally.
            :param in_content: a string containing a edge quantity value
        """   
        for quantity in self.CollectAllMatchesOfPattern(in_content, self.constants.NUMBER_QUANTITIY_REGEX):
            value = int(quantity[2:len(quantity)-2])
            print('V:\n', value)
            fragments = NumWordParser(value).GetDigitsByBase(in_value=value, base=1000)
            fragments_count = len(fragments)
            result = ''

            if(fragments_count == 1 and fragments[0] > 0):
                result = '([low])'
            elif (fragments_count == 2):
                result = '([mid])'
            elif (fragments_count == 3):
                result = '([high])'
            elif (fragments_count >= 4):
                result = '([gigantic])'
            else:
                result = '([zero])'
            in_content = in_content.replace(quantity, result)

        return in_content
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def ExploreAdditionalcontent(self, in_content):
        """
        This function search in a AMR line fragment about additional context for the AMR node.
            :param in_content: a AMR line fragment with a node and maybe additional context 
        """
        try:
            if isInStr(self.constants.COLON, in_content): in_content = self.DeleteFlags(in_content)
            
            if isInStr('-', in_content):
                in_content = self.ReplacePolarity(in_content)
                in_content = self.DeleteWordExtension(in_content)

            in_content = self.RemoveUnusedSignes(in_content)
                    
            return in_content
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.ExploreAdditionalcontent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NiceFormatting(self, in_content):
        """
        This function format and clean up a raw AMR semantic (graph-) string.
        This allow to clean up format definitions and remove [for this project] uninteresting content!
            :param in_content: a raw AMR semantic string
        """
        try:
            depth = -1
            openings = in_content.split(self.node_parenthesis[0])
            struct_contain = []

            for line in openings:
                depth = depth + 1
                new_line = self.AddLeadingSpace((self.node_parenthesis[0] + line), depth)

                if isInStr(self.node_parenthesis[1], new_line):
                    occourences = self.GetSignOccurenceCount(new_line, self.node_parenthesis[1])
                    depth = depth - occourences
                
                new_line = self.ExploreAdditionalcontent(new_line)
                struct_contain.append(new_line)

            return '\n'.join(struct_contain) + ')'

        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.NiceFormatting]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def GenerateCleanAMR(self):
        """
        This function preprocess a raw AMR semantic (graph-) string for further usage in the main project.
        """
        try:
            self.context = self.GetUnformatedAMRString(self.context)
            if  self.HasParenthesis(self.context) and self.MatchSignsOccurences(self.context):
                self.context = self.EncloseUnenclosedValues(self.context)
                self.context = self.EncloseStringifiedValues(self.context)
                self.context = self.GetEnclosedContent(self.context)
                self.context = self.LookUpReplacement(self.context)
                self.cleaned = self.NiceFormatting(self.context)
                self.isCleaned = self.AllowedCharacterOccurenceCheck(self.cleaned)
                if(self.isCleaned):
                    return self.cleaned
                else:
                    return None
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.GenerateCleanAMR]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def AllowedCharacterOccurenceCheck(self, in_context):
        try:
            only_allowed_chars = all(x.isalnum() or x.isspace() or (x is '[') or (x is ']') or (x is '(') or (x is ')')  or (x is '/') or (x is '?') or (x is '\n') for x in in_context)
            has_correct_parenthesis = self.MatchSignsOccurences(in_context) and self.MatchSignsOccurences(in_context, self.edge_parenthesis)
            allowed = only_allowed_chars and has_correct_parenthesis
            return allowed
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.AllowedCharacterOccurenceCheck]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)