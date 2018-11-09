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
    context = None
    cleaned_context = None
    isCleaned = False
    hasContext = False
    hasExtentsionsDict = False
    keep_edge_encoding = False

    # Class inits
    constants = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, open_bracket='(', close_bracket=')', input_context=None, input_extension_dict={}, keep_edges=False):
        try:
            self.constants = Constants()

            if isStr(input_context): 
                self.hasContext = True
                self.context = input_context

            if isDict(input_extension_dict):
                self.hasExtentsionsDict = True
                self.extension_dict = input_extension_dict
                self.extension_keys_dict = self.extension_dict.keys()

            if isStr(open_bracket) and isStr(close_bracket):
                self.node_parenthesis[0] = open_bracket
                self.node_parenthesis[1] = close_bracket

            if isBool(keep_edges): self.keep_edge_encoding = keep_edges
            if(self.hasContext): self.GenerateCleanAMR()

        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.__init__]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def HasColon(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.COLON in in_context
        elif self.hasContext:
            return self.constants.COLON in self.context
        else:
            return False

    def HasQuotation(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.QUOTATION_MARK in in_context
        elif self.hasContext:
            return self.constants.QUOTATION_MARK in self.context
        else:
            return False
    
    def HasParenthesis(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.node_parenthesis[0] in in_context and self.node_parenthesis[1] in in_context
        elif self.hasContext:
            return self.node_parenthesis[0] in self.context and self.node_parenthesis[1] in self.context
        else:
            return False

    def HasWhitspaces(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.WHITESPACE in in_context
        elif self.hasContext:
            return self.constants.WHITESPACE in self.context
        else:
            return False

    def MatchSignsOccurences(self, in_context='', in_signs=['(',')']):
        if isStr(in_context) and len(in_context) > 0:
            count_sign_x = self.CountSignOccurence(in_context, in_signs[0])
            count_sign_y = self.CountSignOccurence(in_context, in_signs[1])
            return count_sign_x == count_sign_y
        elif self.hasContext:
            count_sign_x = self.CountSignOccurence(self.context, in_signs[0])
            count_sign_y = self.CountSignOccurence(self.context, in_signs[1])
            return count_sign_x == count_sign_y
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
    def NewEdge(self , in_context=None):
        """
        This function defines a new AMR converted node edge.
            :param in_context: the corresponding content
        """
        try:
            return self.node_parenthesis[0] + self.edge_parenthesis[0] + in_context + self.edge_parenthesis[1] + self.node_parenthesis[1]
        except ValueError:
            print("ERR: No label passed to [AMRCleaner.NewEdge].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.NewEdge]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)        
        
    def CountLeadingWhiteSpaces(self, in_context):
        """
        This function count leading whitespaces in raw line of a raw or indentation cleaned AMR string.
            :param in_context: a raw or indentation cleaned AMR string.
        """
        try:
            count = 0
            if self.HasWhitspaces(in_context):
                count = (len(in_context) - len(in_context.lstrip(' ')))
            return count
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.CountLeadingWhiteSpaces].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CountLeadingWhiteSpaces]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CountSignOccurence(self, in_context, in_search_element):
        """
        This function count occourences of a string inside another string.
            :param in_context: the string were want to discover occourences inside
            :param in_search_element: the string we are searching for
        """
        try:
            occurence = 0
            if len(in_context) > len(in_search_element):
                occurence = in_context.count(in_search_element)
            return occurence
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.CountSignOccurence].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CountSignOccurence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetNestedContent(self, in_context):
        """
        This function return the most outer nested content in a string by given desired node_parenthesis strings.
            :param in_context: string nested in desired node_parenthesis.
        """
        try:
            if self.HasParenthesis(in_context):
                pos_open = in_context.index(self.node_parenthesis[0])
                pos_close = in_context.rfind(self.node_parenthesis[1])
                in_context = in_context[pos_open+1:pos_close]
            
            return in_context
        except ValueError:
            print("ERR: Missing or wrong value(s) passed to [AMRCleaner.GetNestedContent].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.GetNestedContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RemoveSpacingFormat(self, in_context):
        """
        This function replace all line and space formatting in raw AMR (graph-) string with a single whitespace.
            :param in_context: a raw string from AMR Dataset
        """
        try:
            return (' '.join(in_context.split()))
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.RemoveSpacingFormat]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def CollectAllMatchesOfPattern(self, in_regex_pattern, in_context):
        """
        This function collect all occurences of a regex pattern match.
            :param in_regex_pattern: matcher regex
            :param in_context: input string
        """   
        try:
            return re.findall(in_regex_pattern, in_context)
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.CollectAllMatchesOfPattern].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.CollectAllMatchesOfPattern]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceAllPatternMatches(self, in_pattern_search, in_replace, in_context):
        """
        This function replace a found regex pattern match with a desired replacement value.
            :param in_pattern_search: matcher regex
            :param in_replace: replacement value
            :param in_context: input string
        """   
        try:
            return re.sub(in_pattern_search, in_replace, in_context) 
        except ValueError:
            print("ERR: No content passed to [AMRCleaner.ReplaceAllPatternMatches].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.ReplaceAllPatternMatches]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def AddLeadingSpace(self, depth, in_context):
        """
        This function add a desired amount of whitespaces to a cleaned AMR line fragment.
            :param in_context: the cleaned AMR semantic line fragment
            :param depth: the desired depth rather intended position in the string
        """
        try:
            for _ in range((depth * self.constants.INDENTATION)):
                in_context = ' '+in_context
            return in_context
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.AddLeadingSpace]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncapsulateEdge(self, in_match):
        """
            Encapsulate edge inputs with defined edge parenthesis.
            :param in_match: a edge string component
        """   
        try:
            flag = in_match[0]
            flagged_element = self.edge_parenthesis[0] + in_match[1].lstrip(' ') + self.edge_parenthesis[1]
            return [flag, flagged_element]
        except ValueError:
            print("ERR: Missing or wrong value passed to [AMRCleaner.EncapsulateEdge].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.EncapsulateEdge]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncapsulateUnenclosedValues(self, in_context):
        """
        This function create new AMR nodes on argument flags following unenclosed values.
            :param in_context: a string containing argument flags following unenclosed values
        """
        try:
            if self.HasColon(in_context):
                for loot_elem in self.CollectAllMatchesOfPattern(self.constants.UNENCLOSED_ARGS_REGEX, in_context):
                    search = ''.join(loot_elem)
                    found_flag = loot_elem[0]
                    found_edge = loot_elem[1].lstrip(' ')
                    replace = ''.join([found_flag, ' ' + self.node_parenthesis[0] + found_edge + self.node_parenthesis[1]])

                    if 'ARG' not in loot_elem[0]:
                        if self.keep_edge_encoding:
                            found_flag, found_edge = self.EncapsulateEdge(loot_elem)
                            replace = ''.join([found_flag, ' ' + self.node_parenthesis[0] + found_edge + self.node_parenthesis[1]])
                        else:
                            replace = ''
                        
                    in_context = self.ReplaceAllPatternMatches(search, replace, in_context)
            return in_context
        except ValueError:
            print("ERR: Missing or wrong value passed to [AMRCleaner.EncapsulateUnenclosedValues].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.EncapsulateUnenclosedValues]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncapsulateStringifiedValues(self, in_context):
        """
        This function creates new edge with defined edge_parenthesis for qualified nested values in quotation marks in the input.
            :param in_context: a string with at least one qualified name
        """
        try:
            if self.HasQuotation(in_context):
                for elem in self.CollectAllMatchesOfPattern(self.constants.MARKER_NESTINGS_REGEX, in_context):
                    replacer = ''

                    if hasContent(elem[0]) and elem[0].count(self.constants.QUOTATION_MARK) == 2:
                        if self.keep_edge_encoding:
                            content = re.sub(self.constants.QUOTATION_MARK,'',elem[0]).replace('_',' ')

                            if all(x.isalnum() or x.isspace() for x in content):
                                replacer = self.NewEdge(content)

                    in_context = in_context.replace(elem[0], replacer, 1)
            return in_context
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.EncapsulateStringifiedValues].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.EncapsulateStringifiedValues]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def ReplaceKnownExtensions(self, in_context):
        """
        Replace flags or content with a known entry in the extension dict.
            :param in_context: input string
        """   
        try:
            if ('-' in in_context):
                look_up_control = self.CollectAllMatchesOfPattern(self.constants.EXTENSION_MULTI_WORD_REGEX, in_context)
                if (isNotNone(look_up_control) and isNotNone(self.extension_dict) and isDict(self.extension_dict)):
                    for found in look_up_control:
                        if len(found[0]) > 0 and found[0] in self.extension_keys_dict:
                            in_context = in_context.replace(found[0], self.extension_dict[found[0]])
                    
            return in_context
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.ReplaceKnownExtensions]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplacePolaritySign(self, in_context):
        """
        This function replace negative polarity (-) with a new edge.
            :param in_context: string containing at least on AMR polarity sign
        """
        try:
            replace = ''

            if self.keep_edge_encoding: replace = self.NewEdge(self.constants.NEG_POLARITY)

            in_context = re.sub(self.constants.SIGN_POLARITY_REGEX, replace, in_context)
            return in_context
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.ReplacePolaritySign].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.ReplacePolaritySign]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplacePoliteSign(self, in_context):
        """
        This function replace positive polite (+) signs with a new edge.
            :param in_context: string containing at least on AMR polite sign
        """
        try:
            replace = ''

            if self.keep_edge_encoding: replace = self.NewEdge(self.constants.POS_POLITE)

            in_context = re.sub(self.constants.SIGN_POLITE_REGEX, replace, in_context)
            return in_context
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.ReplacePoliteSign].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.ReplacePoliteSign]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteFlags(self, in_context):
        """
        This function delete AMR flags and only keep the informations they were flagged.
            :param in_context: string with at least on AMR flag
        """
        try:
            if isInStr(self.constants.COLON, in_context):
                in_context = re.sub(self.constants.FLAG_REGEX, '', in_context)
            return in_context
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.DeleteFlags]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteUnusedSigns(self, in_context):
        """
        This function delete all remaining signs we don't want to keep in the string.
            :param in_context: a semantic line fragment
        """
        try:
            return re.sub(self.constants.SIGNS_REMOVE_UNUSED_REGEX, '', in_context)
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.DeleteUnusedSigns].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.DeleteUnusedSigns]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteWordExtension(self, in_context):
        """
        This function delete word extensions from node content in a AMR semantic line fragment.
            :param in_context: a semantic line fragment
        """
        try:
            if isInStr('-', in_context):
                in_context = re.sub(self.constants.EXTENSION_NUMBER_REGEX,'', in_context)
            return in_context
        except ValueError:
            print("ERR: No content passed to [ARMCleaner.DeleteWordExtension].")
        except Exception as ex:
            template = "An exception of type {0} occurred in [ARMCleaner.DeleteWordExtension]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def ExploreAdditionalContent(self, in_context):
        """
        This function search in a AMR line fragment about additional context for the AMR node.
            :param in_context: a AMR line fragment with a node and maybe additional context 
        """
        try:
            if isInStr(self.constants.COLON, in_context): in_context = self.DeleteFlags(in_context)
            
            if isInStr('+', in_context): in_context = self.ReplacePoliteSign(in_context)

            if isInStr('-', in_context):
                in_context = self.ReplacePolaritySign(in_context)
                in_context = self.DeleteWordExtension(in_context)

            in_context = self.DeleteUnusedSigns(in_context)
                    
            return in_context
        except Exception as ex:
            template = "An exception of type {0} occurred in [AMRCleaner.ExploreAdditionalContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NiceFormatting(self, in_context):
        """
        This function format and clean up a raw AMR semantic (graph-) string.
        This allow to clean up format definitions and remove [for this project] uninteresting content!
            :param in_context: a raw AMR semantic string
        """
        try:
            depth = -1
            openings = in_context.split(self.node_parenthesis[0])
            struct_contain = []

            for line in openings:
                depth = depth + 1
                new_line = self.AddLeadingSpace(depth, (self.node_parenthesis[0] + line))

                if isInStr(self.node_parenthesis[1], new_line):
                    occourences = self.CountSignOccurence(new_line, self.node_parenthesis[1])
                    depth = depth - occourences
                
                new_line = self.ExploreAdditionalContent(new_line)
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
            self.context = self.RemoveSpacingFormat(self.context)
            if  self.HasParenthesis(self.context) and self.MatchSignsOccurences(self.context):
                self.context = self.EncapsulateUnenclosedValues(self.context)
                self.context = self.EncapsulateStringifiedValues(self.context)
                self.context = self.GetNestedContent(self.context)
                self.context = self.ReplaceKnownExtensions(self.context)
                self.cleaned_context = self.NiceFormatting(self.context)
                self.isCleaned = self.AllowedCharacterOccurenceCheck(self.cleaned_context)
                if(self.isCleaned):
                    return self.cleaned_context
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