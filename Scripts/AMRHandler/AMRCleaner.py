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
    constants = Constants()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, open_bracket='(', close_bracket=')', input_context=None, input_extension_dict=None):
        try:
            if ((input_context is not None) and isStr(input_context)): 
                self.gotContext = True
                self.context = input_context

            if((input_extension_dict is not None) and isDict(input_extension_dict)):
                self.gotExtentsionsDict = True
                self.extension_dict = input_extension_dict
            #else:
            #    self.extension_dict = 

            if(isStr(open_bracket) and isStr(close_bracket)):
                self.parenthesis[0] = open_bracket
                self.parenthesis[1] = close_bracket

            if(self.gotContext):
                self.cleaned = self.GenerateCleanAMR(input_context, open_bracket, close_bracket)
        except ValueError:
            print("No valid inputs passed. Try again...")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #// TODO BUG: Labels are inserted => (s / string-entity  (Y0Z)) but content is missing (s / string-entity :value "what")
    def CreateNewLabel(self, number):
        """
        This function create a label with a upper-case post- and pre-element.
        The passed number (maybe an iteration value) is placed between them.
            :param number: a number which is placed in the label
        """
        inner_index = GetRandomInt(0, 50)
        if isNumber(number) and (number > -1):
            inner_index = number
            
        return 'Y' + str(inner_index) + 'Z'

    def CreateNewDefinedNode(self, label , content, open_par, close_par):
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

    def CountLeadingWhiteSpaces(self, raw_line):
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

    def GetCurrentDepth(self, raw_line):
        """
        This function calculate the depth of a AMR strig node element depending on:
        1. the amount of leading whitespaces
        2. the global defined constants.INDENTATION constant
            :param raw_line: a raw or indentation cleaned AMR string.
        """
        if isStr(raw_line):
            if isInStr('', raw_line):
                return toInt(self.CountLeadingWhiteSpaces(raw_line) / self.constants.INDENTATION)
            else:
                return 0
        else:
            print('WRONG INPUT FOR [GetCurrentDepth]')
            return None

    def CountSubsStrInStr(self, content_str, search_element):
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

    def CheckOpenEnclosing(self, content, open_par, close_par):
        """
        This function check the equal amount of opened and closed parenthesis depending on the given parenthesis definition.
            :param content: raw string containing desired parenthesis
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        if isStr(content) and isStr(open_par) and isStr(close_par) and isInStr(open_par, content) and isInStr(close_par, content) :
            count_open = self.CountSubsStrInStr(content,open_par)
            count_close = self.CountSubsStrInStr(content,close_par)

            if (count_open == count_close):
                return True
            else:
                return False
        else:
            print('WRONG INPUT FOR [CheckOpenEnclosing]')
            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def GetEnclosedContent(self, content, open_par, close_par):
        """
        This function return the most outer nested content in a string by given desired parenthesis strings.
            :param content: string containing content nested in desired parenthesis.
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        try:
            if isInStr(open_par, content) and isInStr(close_par, content):
                pos_open = content.index(open_par)
                pos_close = content.rfind(close_par)
                return content[pos_open+1:pos_close]
            else:
                return content
        except ValueError:
            print("ERR: Missing or wrong value(s) passed to [GetEnclosedContent].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseSoloLabels(self, raw_line):
        """
        This function create new AMR nodes on argument flags following unenclosed labels.
            :param raw_line: a string containing argument flags following unenclosed labels
        """
        try:
            if isInStr(self.constants.COLON, raw_line):
                loot = re.findall(self.constants.UNENCLOSED_ARGS_REGEX, raw_line)

                for loot_elem in loot:
                    joined_elem_regex = ''.join(loot_elem)
                    joined_elem_replace = ''.join([loot_elem[0], ' ('+loot_elem[1].lstrip(' ')+')'])
                    raw_line = re.sub(joined_elem_regex, joined_elem_replace, raw_line)      
                    
            return raw_line
        except ValueError:
            print("ERR: Missing or wrong value passed to [EncloseSoloLabels].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def EncloseQualifiedStringInforamtions(self, raw_line, open_par, close_par):
        """
        This function creates new AMR nodes with desired parenthesis for Qualified Names enclosed in quotation marks.
            :param raw_line: a string with (at least one) qualified name(s)
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        try:
            if isInStr(self.constants.QUOTATION_MARK, raw_line):
                loot = re.findall(self.constants.QUALIFIED_STR_REGEX, raw_line)
                run_iter = -1

                for loot_elem in loot:
                    found_elem = loot_elem[0]
                    label = None
                    content = None

                    if isNotNone(found_elem) and hasContent(found_elem) and found_elem.count(self.constants.QUOTATION_MARK) == 2:
                        if found_elem in self.new_nodes_dict:
                            label = self.new_nodes_dict[found_elem]
                            content = None

                        else:     
                            run_iter = run_iter + 1
                            label = self.CreateNewLabel(run_iter)
                            content = re.sub(self.constants.QUOTATION_MARK,'',found_elem)
                            self.new_nodes_dict[found_elem] = label
                        
                        replace = self.CreateNewDefinedNode(label, content, open_par, close_par)
                        raw_line = raw_line.replace(found_elem, replace, 1)

                    else:
                        print('WRONG DEFINITION ['+ found_elem + ']FOUND IN INPUT FOR [EncloseQualifiedStringInforamtions]')
                        continue   

            return raw_line
        except ValueError:
            print("ERR: No string passed to [EncloseQualifiedStringInforamtions].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #//TODO Polarity has a to be changed! There still polarity signs occouring => (h / he)  -)! 
    def ReplacePolarity(self, raw_line, open_par, close_par):
        """
        This function replace negative polarity (-) signs in a raw semantic line with a new AMR node.
            :param raw_line: string containing (at least on) AMR polarity sign
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        try:
            if isInStr(' - ', raw_line):
                next_depth = self.GetCurrentDepth(raw_line) + 1 
                label = None
                content = None

                if self.constants.NEG_POLARITY in self.new_nodes_dict:
                    label = self.new_nodes_dict[self.constants.NEG_POLARITY]
                    content = None
                else:
                    label = self.constants.NEG_POL_LABEL
                    content = self.constants.NEG_POLARITY
                    self.new_nodes_dict[self.constants.NEG_POLARITY] = self.constants.NEG_POL_LABEL

                replace_node_str = self.CreateNewDefinedNode(label, content, open_par, close_par)
                replace = self.AddLeadingSpace(replace_node_str, next_depth)
                result = re.sub(self.constants.POLARITY_SIGN_REGEX, ('\n'+ replace), raw_line)
                return result
            else:
                return raw_line
        except ValueError:
            print("ERR: No string passed to [ReplacePolarity].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def DeleteFlags(self, raw_line):
        """
        This function delete AMR flags and only keep the informations they were flagged.
            :param raw_line: string with (at least on) AMR flag(s)
        """
        if isStr(raw_line):
            if isInStr(self.constants.COLON, raw_line):
                return re.sub(self.constants.FLAG_REGEX, '', raw_line)            
            else:
                return raw_line
        else:
            print('WRONG INPUT FOR [DeleteFlags]')
            return None

    def RemoveUnusedSignes(self, raw_line):
        """

        """
        if isStr(raw_line):
                return re.sub(self.constants.REMOVE_USELSS_ELEMENTS_REGEX, '', raw_line)            
        else:
            return raw_line

    #//TODO BUG nicht alle extensions werden gelöscht => (l / long-03)
    #//TODO how to handle => :part-of(x) if x is a parent?
    #//TODO how to handle => amr-unknown content? Its dataset content!
    #//TODO how to handle => toss-out and have-rel-role? Both types occour much often! ??? mglw. => einzelworte in Glove später addieren?
    #//TODO how to handle => :mode <random word>? I actually replace them as new node but the label ist still missing for it! mglw. => label = word[0] + 0 + word[0]?
    def DeleteWordExtension(self, raw_line):
        """
        This function delete word extensions from node content in a AMR semantic line fragment.
            :param raw_line: a AMR semantic line fragment
        """
        if isStr(raw_line):
            if isInStr('-', raw_line):
                return re.sub(self.constants.EXTENSION_ELEMENT_REGEX,'', raw_line)
            else:
                return raw_line
        else:
            print('WRONG INPUT FOR [DeleteWordExtension]')
            return None
        
    def ExploreAdditionalcontent(self, raw_line, open_par, close_par):
        """
        This function search in a AMR line fragment about additional context for the AMR node.
            :param raw_line: a AMR line fragment with a node and maybe additional context 
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        if isStr(raw_line):
            result = raw_line

            if isInStr(self.constants.COLON, result):
                result = self.DeleteFlags(result)
            if isInStr('-', result):
                result = self.ReplacePolarity(result, open_par, close_par)
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

    def LookUpReplacement(self, raw_line):
        try:
            if ('-' in raw_line) and (re.findall(self.constants.EXTENSION_REGEX, raw_line) is not None):
                for found in re.findall(self.constants.EXTENSION_REGEX, raw_line):
                    print('[',found,']')
                #extension_dict
            
            return raw_line
        except ValueError:
            print("ERR: No string passed to [LookUpReplacement].")
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
                    occourences = self.CountSubsStrInStr(new_line, close_par)
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

    def GenerateCleanAMR(self, raw_amr, open_par, close_par):
        """
        This function preprocess a raw AMR semantic (graph-) string for further usage in the main project.
            :param raw_amr: a raw AMR semantic (graph-) string
            :param open_par: string of desired parenthesis type style "open"
            :param close_par: string of desired parenthesis type style "close"
        """
        if isStr(raw_amr) and isStr(open_par) and isStr(close_par):
            unformated_str = self.GetUnformatedAMRString(raw_amr)
            if self.CheckOpenEnclosing(unformated_str, open_par, close_par):
                node_enclosed_str = self.EncloseSoloLabels(unformated_str)
                name_enclosed_str = self.EncloseQualifiedStringInforamtions(node_enclosed_str, open_par, close_par)
                amr_str = self.GetEnclosedContent(name_enclosed_str, open_par, close_par)
                look_str = self.LookUpReplacement(amr_str)
                result = self.NiceFormatting(look_str, open_par, close_par)
                print(result)
                
                """
                #//TODO INFO: this control structure check extension regex failed sometimes!
                if '-' in result:
                    if (re.findall(self.constants.POLARITY_SIGN_REGEX, result) is not None):
                        for found in re.findall(self.constants.POLARITY_SIGN_REGEX, result):
                            self.extension_dict['EXT>'+found[0]] = found[0]

                    if (re.findall(self.constants.EXTENSION_REGEX, result) is not None):
                        for found in re.findall(self.constants.EXTENSION_REGEX, result):
                            self.extension_dict['EXT>'+found[0]] = found[0]
                """
                return result
            else:
                print('UNEQUAL AMOUNT OF BRACKET PAIRS IN INPUT FOR [GenerateCleanAMR]')
                return None
        else:
            print('WRONG INPUT FOR [GenerateCleanAMR]')
            return None

'''
        try:
            
        except ValueError:
            print("ERR: No string passed to [].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
'''