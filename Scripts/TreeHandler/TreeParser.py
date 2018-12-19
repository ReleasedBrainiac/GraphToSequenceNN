# - *- coding: utf-8*-
'''
    Used Resources:
        => http://anytree.readthedocs.io/en/latest/
'''

import re
from anytree import AnyNode, RenderTree, find, findall, PreOrderIter
from collections import OrderedDict
from DatasetHandler.ContentSupport import isDict, isAnyNode, isStr, isInt, isODict, isList, isBool, isNone, isNotEmptyString
from DatasetHandler.ContentSupport import isInStr, isNotInStr, isNotNone, toInt
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from Configurable.ProjectConstants import Constants

'''
    This class library allow to extract content from AMR String representation.
    The result will provide as AnyNode or Json-AnyNode representation.
'''
class TParser:

    #TODO add/edit documentation remove unclean comments

    constants = None
    amr_input = None
    show = False
    saving = False


    def __init__(self, in_amr_stringified, show_process, is_saving):
        """
        This class constructor only collect necessary inputs and initialize the constants.
            :param in_amr_stringified: amr input as string
            :param show_process: switch allow to print some processing steps
            :param is_saving: switch allow to show further passing strategy after processing data
        """   
        try:
            self.constants = Constants()
            self.amr_input = in_amr_stringified
            self.show = show_process
            self.saving = is_saving
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExportToJson(self, anytree_root):
        """
        This function allow to convert a AnyNode Tree to a AnyNode-JsonString representation.
            :param anytree_root: root of an anytree object
        """   
        try:
            exporter = JsonExporter(indent=2, sort_keys=True)
            return '#::smt\n' + exporter.export(anytree_root) + '\n'
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ExportToJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ImportAsJson(self, json):
        """
        This function allow to convert a AnyNode-JsonString representation to a AnyNode Tree.
            :param json: json string defining a anytree
        """   
        try:
            importer = JsonImporter()
            return importer.import_(json)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ImportAsJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractRawNodeSequence(self, sequence):
        """
        This function allow to collect the raw node sequence containing the label 
        and maybe some additional values like flags and description content.
            :param sequence: node sequence string
        """   
        try:
            node_sentence = sequence.lstrip(' ')+')'
            return node_sentence[node_sentence.find("(")+1:node_sentence.find(")")] 
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ExtractRawNodeSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==               Get nodes label and content             ==#
    # This function allow to collect the Label and eventually a label content 
    # from string of a AMR Graph variable.
    def GetSplittedContent(self, input_node):
        """
        This function allow to collect the label or the full node definition 
        from amr substring variable.
            :param input_node: amr substring containing at least just 1 node.
        """   
        try:
            parts = input_node.lstrip(' ').split('/')
            if len(parts) == 1:
                return parts, None
            else:
                label = parts[0]
                content = parts[1]
                return label, content
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetSplittedContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==                    Cleanup AMR spacing                ==#
    # This function clean up the whitespacing in the AMR Graphstring by a given value.
    def AddLeadingWhitespaces(self, str, amount):
        try:
            for _ in range(amount):
                str = ' '+str

            ws_count = len(str) - len(str.lstrip(' '))
            return [str, ws_count]
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.AddLeadingWhitespaces]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CleanSubSequence(self, elements):
        try:
            results = []

            if isNotNone(elements) and isList(elements):
                # Clean the content
                for value in elements:
                    if(isInStr("-", value)) or (isInStr(":", value)):
                        if(isInStr("-", value)) and (isNotInStr(":", value)):
                            str1 = value[0: value.rfind('-')]
                            if(len(str1) > 0):
                                results.append(str1)
                        else:
                            continue
                    else:
                        if(len(value) > 0):
                            results.append(value)
            else:
                print('No content is given!')

            return results
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.CleanSubSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==          Cleaning nodes sequence from garbage         ==#
    # This function allow to clean the node sequence from all staff so it return label and content only.
    # It will also cut of word extensions so we just get the basis word of a nodes content!
    def CleanNodeSequence(self, sequence):
        try:
            node_seq = self.ExtractRawNodeSequence(sequence)
            # If we have more then just a label
            if(isInStr(' ', node_seq)):
                elements = node_seq.split(' ')
                results = self.CleanSubSequence(elements)
                node_seq = ' '.join(results)

            return node_seq
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.CleanNodeSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==             Show RLDAG tree representation            ==#
    # This function show a rendered representation of a AnyNode tree on console!
    def ShowRLDAGTree(self, root):
        try:
            print(RenderTree(root))
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ShowRLDAGTree]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==               Search for depending parent             ==#
    # This function allow to get the parent node of the given node depending on a given depth.
    # The depth is needed to find the correct node during navigation through a graph construction.
    # We search with the previous node and define the steps we have to step up in the tree. 
    def GetParentWithPrev(self, node, cur_depth):
        try:
            cur_parent = node.parent
            while((cur_parent.depth + 1) > cur_depth):
                cur_parent = cur_parent.parent

            return cur_parent
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetParentWithPrev]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==               Search for new node parent              ==#
    # This function allow to get the parent for a new node depending on:
    # 1. depth of the new node
    # 2. depth of the previous inserted node in the tree
    # 3. previous AnyNode element at insertion in the tree
    # This method is used in the BuildNextNode function only because its a part of the workaround!
    def GetParentOfNewNode(self, depth, p_depth, prev_node):
        try:
            #Next step down in the Rooted Label DAG (RLDAG)
            if(depth > p_depth):
                return prev_node
            else:
                #Next same layer Node in the RLDAG
                if(depth == p_depth):
                    return prev_node.parent
                #Next rising layer Node in the RLDAG
                else:
                    return self.GetParentWithPrev(prev_node, depth)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetParentOfNewNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildTreeGraph(self, orderedNodesDepth, orderedNodesContent):
        """
        This function build a graph like tree from (R)outed (L)abeled (D)irected (A)cyclic (G)raph [RLDAG].
            :param orderedNodesDepth: list of nodes depths in order of occurrence
            :param orderedNodesContent: list of nodes labels and featues in order of occurrence
        """   
        try:
            if (isODict(orderedNodesDepth)) and (isODict(orderedNodesContent)):
                if(len(orderedNodesDepth) != len(orderedNodesContent)): return None
                else:
                    root = None
                    prev_index = -1
                    prev_node = None
                    depth = None
                    label = None
                    content = None

                    for index in range(len(orderedNodesDepth)):
                        depth = orderedNodesDepth[index]
                        label, content = self.GetSplittedContent(orderedNodesContent[index])

                        if(index == 0):
                            root = self.NewAnyNode( index, 'root', depth, False, [], True, [], label, content)
                            prev_node = root
                        else:
                            parent_depth = orderedNodesDepth[prev_index]
                            prev_node = self.BuildNextNode( prev_node, index, parent_depth, None, depth, True, [], False, [], label, content)

                        prev_index = index
                return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.BuildTreeGraph]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NewAnyNode(self, node_id, node_State, node_Depth, node_HasInputNode, node_InputNode, node_HasFollowerNodes, node_FollowerNodes, node_Label, node_Content):
        """
        This function create a new node containing the given params.
            :param node_id: node id
            :param node_State: node type
            :param node_Depth: node position in depth
            :param node_HasInputNode: parent check for at least 1
            :param node_InputNode: list of parents
            :param node_HasFollowerNodes: children check for at least 1
            :param node_FollowerNodes: list of children
            :param node_Label: obviously the node label
            :param node_Content: the features stored in the node
        """   
        try:
            if(node_HasInputNode == False): node_InputNode = None

            return AnyNode( id=node_id,
                            name=node_State,
                            state=node_State,
                            depth=node_Depth,
                            hasInputNode=node_HasInputNode,
                            parent=node_InputNode,
                            hasFollowerNodes=node_HasFollowerNodes,
                            followerNodes=node_FollowerNodes,
                            label=node_Label,
                            content=node_Content)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.NewAnyNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildNextNode(self, prev_node, index, p_depth, state, depth, hasInputs, input, hasFollowers, followers, label, content):
        """
        This function create a new node depending on a given node or tree.
        Here the node gets its position inside the tree depending on the given prev node which is the root or a tree structure.
            :param prev_node: 
            :param index: 
            :param p_depth: 
            :param state: 
            :param depth: 
            :param hasInputs: 
            :param input: 
            :param hasFollowers: 
            :param followers: 
            :param label: 
            :param content: 
        """   
        try:
            if(index > 0):
                input = self.GetParentOfNewNode(depth, p_depth, prev_node)
                prev_node = self.NewAnyNode(index,
                                            state,
                                            depth,
                                            hasInputs,
                                            input,
                                            hasFollowers,
                                            followers,
                                            label,
                                            content)

            return prev_node
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.BuildNextNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NormalState(self, node):
        """
        This funtion set the state of a node in the GraphTree.
            :param node: 
        """   
        try:
            if (node.is_leaf) and not (node.is_root):
                node.state = 'destination'
                node.name = 'destination_' + str(node.id)
            elif(node.is_root):
                node.state = 'root'
                node.name = 'root_' + str(node.id)
                node.parent = None
            else:
                node.state = 'subnode'
                node.name = 'subnode_' + str(node.id)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.NormalState]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NavigateState(self, graph_root, node):
        """
        This function chech notes if they are root, subnode or child (-> DAG review as Tree)
        If this check is true it calls NormalState function.
        Otherwise it sets state to 'navigator'.
        Graph of Rank 1 is a sigle root node!
            :param graph_root: 
            :param node: 
        """   
        try:
            if isNotNone(node.label) and isNone(node.content):
                label = node.label
                regex = str('\\b'+label+'\\b')
                desired = []

                tmp_desired =findall(graph_root, lambda node: node.label in label)

                for i in tmp_desired:
                    match = re.findall(regex, i.label)
                    if len(match) > 0:
                        desired.append(i)

                if(len(desired) < 1):
                    print( node.state )
                elif(len(desired) == 1):
                    self.NormalState(node)
                else:
                    node.followerNodes = desired[0].followerNodes
                    node.hasFollowerNodes = desired[0].hasFollowerNodes
                    node.hasInputNode = desired[0].hasInputNode
                    node.state = 'navigator'
                    node.name = 'navigator_' + str(node.id)
            else:
                self.NormalState(node)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.NavigateState]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetSubnodes(self, node):
        """
        This function place sub nodes to the current node if some exist.
            :param node: 
        """   
        try:
            followers = [node.label for node in node.children]
            if(len(followers) > 0):
                node.hasFollowerNodes = True
                node.followerNodes = followers
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetSubnodes]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GraphReforge(self, root):
        """
        This function wrap the SingleNodeReforge for all GraphTree nodes.
            :param root: 
        """
        try:
            for node in PreOrderIter(root):
                self.NavigateState(root, node)
                self.GetSubnodes(node)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GraphReforge]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ShowGathererInfo(self, amr_stringified, root):
        """
        This function print the input and result of the gatherer.
        This allow to evaluate the gatherers work easily.
            :param amr_stringified: 
            :param root: 
        """   
        try:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(amr_stringified)
            self.ShowRLDAGTree(root)
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ShowGathererInfo]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Preprocessor(self, graph_nodes, nodes_depth, nodes_content):
        """
        docstring here
            :param graph_nodes: 
            :param nodes_depth: 
            :param nodes_content: 
        """   
        try:
            depth = -1
            k = 0
            for line in graph_nodes:
                if isNotEmptyString(line):
                    if(self.constants.SEMANTIC_DELIM not in line):
                        depth = depth + line.count('(')
                        nodes_depth[k] = depth
                        nodes_content[k] = self.CleanNodeSequence(line)
                        depth = depth - line.count(')')
                        k = k + 1
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Preprocessor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Pipeline(self, amr_stringified, semantic_flag, print_to_console):
        """
        This function gather a TreeGraph as AnyNode Tree from AMR-String-Representation.
            :param amr_stringified: 
            :param semantic_flag: 
            :param print_to_console: 
        """   
        try:
            nodes_depth = OrderedDict()
            nodes_content = OrderedDict()
            self.Preprocessor(amr_stringified.split('\n'), nodes_depth, nodes_content)
            root = self.BuildTreeGraph(nodes_depth, nodes_content)
            self.GraphReforge(root)
            if(print_to_console): self.ShowGathererInfo(amr_stringified, root)
            return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Execute(self):
        """
        This function handle the execution of the gatherer.
        The result is a importable json if you like to store [to_process = False] 
        or a AnyNode Tree if you want to process it further [to_process = True]
        """   
        try:
            root = self.Pipeline(self.amr_input, self.constants.SEMANTIC_DELIM, self.show)
            if(self.saving): return self.ExportToJson(root)
            else: return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)