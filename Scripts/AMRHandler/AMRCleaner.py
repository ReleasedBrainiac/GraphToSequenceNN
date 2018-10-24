# - *- coding: utf-8*-
import re
from DatasetHandler.ContentSupport import isInStr, isNotInStr, isNotNone, isStr, isInt, toInt, isNumber, isDict
from DatasetHandler.ContentSupport import hasContent, GetRandomInt
from Configurable.ProjectConstants import Constants

class Cleaner:
    #//TODO IDEE: Erst den String in einen Baum packen und dann die einzelnen Knoten bearbeiten! Das macht den ganzen Vorgang einfacher!

    # Variables inits
    parenthesis = ['(',')']
    extension_dict = {}
    new_nodes_dict = {}
    context = None
    cleaned = None
    isCleaned = False
    gotContext = False
    gotExtentsionsDict = False

    # Class inits
    constants = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, open_bracket='(', close_bracket=')', input_context=None, input_extension_dict={}):
        try:

            self.constants = Constants()
            self.new_nodes_dict = {}

            if isStr(input_context): 
                self.gotContext = True
                self.context = input_context

            if isDict(input_extension_dict):
                self.gotExtentsionsDict = True
                self.extension_dict = input_extension_dict

            if isStr(open_bracket) and isStr(close_bracket):
                self.parenthesis[0] = open_bracket
                self.parenthesis[1] = close_bracket

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
            return self.parenthesis[0] in in_context and self.parenthesis[1] in in_context
        elif self.gotContext:
            return self.parenthesis[0] in self.context and self.parenthesis[1] in self.context
        else:
            return False

    def HasPolarity(self, in_context=''):
        if isStr(in_context) and len(in_context) > 0:
            return self.constants.POLARITY in in_context
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

    def MatchSignsOccurences(self, in_context='', in_sign_x='(', in_sign_y=')'):
        if isStr(in_context) and len(in_context) > 0:
            count_sign_x = self.GetSignOccurenceCount(in_context, in_sign_x)
            count_sign_y = self.GetSignOccurenceCount(in_context, in_sign_y)
            return count_sign_x == count_sign_y
        elif self.gotContext:
            count_sign_x = self.GetSignOccurenceCount(self.context, in_sign_x)
            count_sign_y = self.GetSignOccurenceCount(self.context, in_sign_y)
            return count_sign_x == count_sign_y
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def CreateNewLabel(self, in_number):
        """
        This function create a label with a upper-case post- and pre-element.
        The passed number (maybe an iteration value) is placed between them.
            :param in_number: a number which is placed in the label
        """
        try:
            inner_index = GetRandomInt(0, 50)
            if isNumber(in_number) and (in_number > -1):
                inner_index = in_number
                
            return 'Y' + str(inner_index) + 'Z'
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
 
    def CreateNewNode(self, in_label , in_content=None):
        """
        This function defines a new AMR converted node element.
        Depending on the content it returns a node with label and content or just containing a label.
            :param in_label: the desired node label
            :param in_content: the corresponding content
        """
        try:
            if isStr(in_content):
                return self.parenthesis[0] + in_label + ' / ' + in_content + self.parenthesis[1]
            else:
                return self.parenthesis[0] + in_label + self.parenthesis[1]
        except ValueError:
            print("ERR: No label passed to [CreateNewNode].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)        
        
    def CountLeadingWhiteSpaces(self, in_content):
        """
        This function count leading whitespaces in raw line of a raw or indentation cleaned AMR string.
        Remind the input should not be preformated (e.g. cleaning from whitespaces).
            :param in_content: a raw or indentation cleaned AMR string.
        """
        try:
            if self.HasWhitspaces(in_content):
                return (len(in_content) - len(in_content.lstrip(' ')))
            else:
                return 0
        except ValueError:
            print("ERR: No content passed to [CountLeadingWhiteSpaces].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetSignOccurenceCount(self, in_content, in_search_element):
        """
        This function count occourences of srtings inside another string.
            :param in_content: the string were want to discover occourences
            :param in_search_element: the string we are searching for
        """
        try:
            if len(in_content) > len(in_search_element):
                return in_content.count(in_search_element)
            else: 
                print('WRONG INPUT FOR [GetSignOccurenceCount]')
                return 0
        except ValueError:
            print("ERR: No content passed to [GetSignOccurenceCount].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
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
            if self.HasWhitspaces(in_content):
                return toInt(self.CountLeadingWhiteSpaces(in_content) / self.constants.INDENTATION)
            else:
                return 0
        except ValueError:
            print("ERR: No content passed to [GetCurrentDepth].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetEnclosedContent(self, in_content):
        """
        This function return the most outer nested content in a string by given desired parenthesis strings.
            :param in_content: string containing content nested in desired parenthesis.
        """
        try:
            if self.HasParenthesis(in_content):
                pos_open = in_content.index(self.parenthesis[0])
                pos_close = in_content.rfind(self.parenthesis[1])
                return in_content[pos_open+1:pos_close]
            else:
                return in_content
        except ValueError:
            print("ERR: Missing or wrong value(s) passed to [GetEnclosedContent].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def CollectAllMatchesOfPattern(self, in_context, in_regex_pattern):
        try:
            return re.findall(in_regex_pattern, in_context)
        except ValueError:
            print("ERR: No content passed to [CollectAllMatchesOfPattern].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ReplaceAllPatternMatches(self, in_pattern_search, in_replace, in_context):
        try:
            return re.sub(in_pattern_search, in_replace, in_context) 
        except ValueError:
            print("ERR: No content passed to [ReplaceAllPatternMatches].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def EncloseSoloLabels(self, in_content):
        """
        This function create new AMR nodes on argument flags following unenclosed labels.
            :param in_content: a string containing argument flags following unenclosed labels
        """
        try:
            if self.HasColon(in_content):
                for loot_elem in self.CollectAllMatchesOfPattern(self.constants.UNENCLOSED_ARGS_REGEX, in_content):
                    search_pattern = ''.join(loot_elem)
                    replace_pattern = ''.join([loot_elem[0], ' ('+loot_elem[1].lstrip(' ')+')'])
                    in_content = self.ReplaceAllPatternMatches(search_pattern, replace_pattern, in_content)
                    
            return in_content
        except ValueError:
            print("ERR: Missing or wrong value passed to [EncloseSoloLabels].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseFreeInformations(self, in_content):
        """
        This function creates new AMR nodes with desired parenthesis for Qualified Names enclosed in quotation marks.
            :param in_content: a string with (at least one) qualified name(s)
        """
        try:
            if self.HasQuotation(in_content):
                run_iter = -1

                for loot_elem in self.CollectAllMatchesOfPattern(in_content, self.constants.QUALIFIED_STR_REGEX):
                    found_elem = loot_elem[0]
                    label = None
                    content = None

                    #//TODO BUG: Hier kontrollieren das der Knotenwerk min einmal im graphen auftaucht 
                    #            oder kein globales sondern ein lokales dict!
                    if hasContent(found_elem) and found_elem.count(self.constants.QUOTATION_MARK) == 2:
                        if (found_elem in self.new_nodes_dict):
                            label = self.new_nodes_dict[found_elem]
                            content = None
                        else:     
                            run_iter = run_iter + 1
                            label = self.CreateNewLabel(run_iter)
                            content = re.sub(self.constants.QUOTATION_MARK,'',found_elem)
                            self.new_nodes_dict[found_elem] = label
                        
                        in_content = in_content.replace(found_elem, self.CreateNewNode(label, content), 1)

                    else:
                        print('WRONG DEFINITION ['+ found_elem + ']FOUND IN INPUT FOR [EncloseFreeInformations]')
                        continue   

            return in_content
        except ValueError:
            print("ERR: No content passed to [EncloseFreeInformations].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #//TODO BUG: Polarity has to be changed! There still polarity signs occouring => (h / he)  -)! 
    def ReplacePolarity(self, in_content):
        """
        This function replace negative polarity (-) signs in a raw semantic line with a new AMR node.
            :param in_content: string containing (at least on) AMR polarity sign
        """
        try:
            if self.HasPolarity(in_content):
                next_depth = self.GetCurrentDepth(in_content) + 1 
                label = None
                content = None

                if (self.constants.NEG_POLARITY in self.new_nodes_dict):
                    label = self.new_nodes_dict[self.constants.NEG_POLARITY]
                    content = None
                else:
                    label = self.constants.NEG_POL_LABEL
                    content = self.constants.NEG_POLARITY
                    self.new_nodes_dict[self.constants.NEG_POLARITY] = self.constants.NEG_POL_LABEL

                replace_node_str = self.CreateNewNode(label, content)
                replace = self.AddLeadingSpace(replace_node_str, next_depth)
                result = re.sub(self.constants.POLARITY_SIGN_REGEX, ('\n'+ replace), in_content)
                return result
            else:
                return in_content
        except ValueError:
            print("ERR: No content passed to [ReplacePolarity].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteFlags(self, in_content):
        """
        This function delete AMR flags and only keep the informations they were flagged.
            :param in_content: string with (at least on) AMR flag(s)
        """
        if isStr(in_content):
            if isInStr(self.constants.COLON, in_content):
                return re.sub(self.constants.FLAG_REGEX, '', in_content)            
            else:
                return in_content
        else:
            print('WRONG INPUT FOR [DeleteFlags]')
            return None

    def RemoveUnusedSignes(self, in_content):
        """

        """
        if isStr(in_content):
                return re.sub(self.constants.REMOVE_USELSS_ELEMENTS_REGEX, '', in_content)            
        else:
            return in_content

    #//TODO BUG nicht alle extensions werden gelöscht => (l / long-03)
    #//TODO how to handle => :part-of(x) if x is a parent?
    #//TODO how to handle => amr-unknown content? Its dataset content!
    #//TODO how to handle => toss-out and have-rel-role? Both types occour much often! ??? mglw. => einzelworte in Glove später addieren?
    #//TODO how to handle => :mode <random word>? I actually replace them as new node but the label ist still missing for it! mglw. => label = word[0] + 0 + word[0]?
    def DeleteWordExtension(self, in_content):
        """
        This function delete word extensions from node content in a AMR semantic line fragment.
            :param in_content: a AMR semantic line fragment
        """
        if isStr(in_content):
            if isInStr('-', in_content):
                return re.sub(self.constants.EXTENSION_ELEMENT_REGEX,'', in_content)
            else:
                return in_content
        else:
            print('WRONG INPUT FOR [DeleteWordExtension]')
            return None
        
    def ExploreAdditionalcontent(self, in_content, open_par, close_par):
        """
        This function search in a AMR line fragment about additional context for the AMR node.
            :param in_content: a AMR line fragment with a node and maybe additional context 
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        if isStr(in_content):
            result = in_content

            if isInStr(self.constants.COLON, result):
                result = self.DeleteFlags(result)
            if isInStr('-', result):
                result = self.ReplacePolarity(result)
                result = self.DeleteWordExtension(result)
                result = self.RemoveUnusedSignes(result)
                    
            return result
        else:
            print('WRONG INPUT FOR [ExploreAdditionalcontent]')
            return None

    def GetUnformatedAMRString(self, raw_amr):
        """
        This function replace all line and space formatting in raw AMR (graph-) string with a sigle whitespace.
            :param raw_amr: a raw AMR (graph-) string from AMR Dataset
        """
        if isStr(raw_amr):
            return (' '.join(raw_amr.split()))
        else:
            print('WRONG INPUT FOR [GetUnformatedAMRString]')
            return None

    def AddLeadingSpace(self, str, depth):
        """
        This function add a desired depth to a cleaned AMR line fragment by means of leading whitespaces.
        This will keep the AMR datasets indentation definition of a AMR semantice (graph) string.
            :param str: the cleaned AMR semantic line fragment
            :param depth: the desired depth rather intended position in the string
        """
        if(isStr(str)) and (isInt(depth)):
            for _ in range((depth * self.constants.INDENTATION)):
                str = ' '+str
            return str

        else:
            print('WRONG INPUT FOR [AddLeadingSpace]')
            return None

    def LookUpReplacement(self, in_content):
        try:
            if ('-' in in_content) and (re.findall(self.constants.EXTENSION_REGEX, in_content) is not None):
                for found in re.findall(self.constants.EXTENSION_REGEX, in_content):
                    print('[',found,']')
                #extension_dict
            
            return in_content
        except ValueError:
            print("ERR: No content passed to [LookUpReplacement].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def NiceFormatting(self, amr_str, open_par, close_par):
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
                new_line = self.AddLeadingSpace((open_par + line), depth)
                #print('[',new_line,']')

                if isInStr(close_par, new_line):
                    occourences = self.GetSignOccurenceCount(new_line, close_par)
                    depth = depth - occourences
                
                if isInStr(self.constants.COLON, new_line):
                    new_line = self.ExploreAdditionalcontent(new_line, open_par, close_par)
                
                struct_contain.append(new_line)

            returning = '\n'.join(struct_contain) + ')'
            return returning

        else:
            print('WRONG INPUT FOR [NiceFormatting]')
            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def GenerateCleanAMR(self):
        """
        This function preprocess a raw AMR semantic (graph-) string for further usage in the main project.
        """
        self.context = self.GetUnformatedAMRString(self.context)
        if  self.HasParenthesis(self.context) and self.MatchSignsOccurences(self.context, self.parenthesis[0], self.parenthesis[1]):
            self.context = self.EncloseSoloLabels(self.context)
            self.context = self.EncloseFreeInformations(self.context)
            self.context = self.GetEnclosedContent(self.context)
            self.context = self.LookUpReplacement(self.context)
            self.cleaned = self.NiceFormatting(self.context, self.parenthesis[0], self.parenthesis[1])

            print(self.cleaned)
            #self.Check(self.cleaned)

            #//TODO hier muss eine Kontrollstruktur rein die einen aussage darüber trifft ob das resultat valide ist.
            #       das resultat muss dann zur eliminierung von fehlerhaften paaren genutzt werden

            return self.cleaned
        else:
            print('UNEQUAL AMOUNT OF BRACKET PAIRS IN INPUT FOR [GenerateCleanAMR]')
            return None

    def Check(self, in_context):
        #//TODO INFO: this control structure check extension regex failed sometimes!
        if '-' in in_context:
            if (re.findall(self.constants.POLARITY_SIGN_REGEX, in_context) is not None):
                for found in re.findall(self.constants.POLARITY_SIGN_REGEX, in_context):
                    self.extension_dict['EXT>'+found[0]] = found[0]

            if (re.findall(self.constants.EXTENSION_REGEX, in_context) is not None):
                for found in re.findall(self.constants.EXTENSION_REGEX, in_context):
                    self.extension_dict['EXT>'+found[0]] = found[0]

'''
        try:
            
        except ValueError:
            print("ERR: No content passed to [].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
'''