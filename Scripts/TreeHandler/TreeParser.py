# - *- coding: utf-8*-
'''
    Used Resources:
        => http://anytree.readthedocs.io/en/latest/
'''

import re
from anytree import AnyNode, RenderTree, find, findall, PreOrderIter
from collections import OrderedDict
from DatasetHandler.ContentSupport import isDict, isAnyNode, isStr, isInt, isODict, isList, isBool, isNone
from DatasetHandler.ContentSupport import isInStr, isNotInStr, isNotNone, toInt
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from Configurable.ProjectConstants import Constants

'''
    This class library allow to extract content from AMR String representation.
    The result will provide as AnyNode or Json-AnyNode representation.
'''
class TParser:

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
            template = "An exception of type {0} occurred in [TreeParser.Constructor]. Arguments:\n{1!r}"
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
            template = "An exception of type {0} occurred in [TreeParser.ExportToJson]. Arguments:\n{1!r}"
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
            template = "An exception of type {0} occurred in [TreeParser.ImportAsJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractRawNodeSequence(self, sequence):
        """
        This function allow to collect the raw node sequence containing the label 
        and maybe some additional values like flags and description content.
            :param sequence: node sequence string
        """   
        try:
            node_sent = re.sub(' +',' ',sequence+')')
            result = node_sent[node_sent.find("(")+1:node_sent.find(")")]
            if isInStr("(", result) or isInStr(")", result):
                print("ERROR: ", result)
            return result 
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.ExtractRawNodeSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==               Get nodes label and content             ==#
    # This function allow to collect the Label and eventually a label content 
    # from string of a AMR Graph variable.
    def GetSplittedContent(self, str):
        """
        This function allow to collect the label or the full node definition 
        from amr substring variable.
            :param str: amr substring containing at least just 1 node.
        """   
        try:
            if isNotInStr('/', str):
                return str, None
            else:
                parts = str.split('/')
                label = parts[0].replace(" ", "")
                content = parts[1].replace(" ", "")
                return label, content
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.GetSplittedContent]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==                    Cleanup AMR spacing                ==#
    # This function clean up the whitespacing in the AMR Graphstring by a given value.
    def AddLeadingWhitespaces(self, str, amount):
        if(isStr(str)) and (isInt(amount)):
            for _ in range(amount):
                str = ' '+str

            ws_count = len(str) - len(str.lstrip(' '))
            return [str, ws_count]

        else:
            print('WRONG INPUT FOR [AddLeadingWhitespaces]')
            return None

    def CleanSubSequence(self, elements):
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

    #==          Cleaning nodes sequence from garbage         ==#
    # This function allow to clean the node sequence from all staff so it return label and content only.
    # It will also cut of word extensions so we just get the basis word of a nodes content!
    def CleanNodeSequence(self, sequence):
        if (isStr(sequence)):
            node_seq = self.ExtractRawNodeSequence(sequence)
            # If we have more then just a label
            if(isInStr(' ', node_seq)):
                elements = node_seq.split(' ')
                results = self.CleanSubSequence(elements)
                node_seq = ' '.join(results)
            else:   
                # If we just have label
                node_seq = node_seq[0: node_seq.rfind('-')]

            return node_seq
        else:
            print('WRONG INPUT FOR [CleanNodeSequence]')
            return None

    #==             Show RLDAG tree representation            ==#
    # This function show a rendered representation of a AnyNode tree on console!
    def ShowRLDAGTree(self, root):
        if(isAnyNode(root)):
            print(RenderTree(root))
        else:
            print('WRONG INPUT FOR [ShowRLDAGTree]')

    #==               Search for depending parent             ==#
    # This function allow to get the parent node of the given node depending on a given depth.
    # The depth is needed to find the correct node during navigation through a graph construction.
    # We search with the previous node and define the steps we have to step up in the tree. 
    def GetParentWithPrev(self, node, cur_depth):
        if isAnyNode(node) and isInt(cur_depth) and isNotNone(cur_depth):
            cur_parent = node.parent
            while((cur_parent.depth + 1) > cur_depth):
                cur_parent = cur_parent.parent

            return cur_parent
        else:
            print('WRONG INPUT FOR [GetParentWithPrev]')
            return None

    #==               Search for new node parent              ==#
    # This function allow to get the parent for a new node depending on:
    # 1. depth of the new node
    # 2. depth of the previous inserted node in the tree
    # 3. previous AnyNode element at insertion in the tree
    # This method is used in the BuildNextNode function only because its a part of the workaround!
    def GetParentOfNewNode(self, depth, p_depth, prev_node):
        if isInt(depth) and isInt(p_depth) and isAnyNode(prev_node):
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
        else:
            print('WRONG INPUT FOR [GetParentOfNewNode]')
            return None

    #==          Create edge-matrix from GraphTree            ==#
    def GetEdgeMatrixFromGraphTree(self):
        print('The method [GetEdgeMatrixFromGraphTree] is currently not implemented!')
        return None

    #==          Create edge-matrix from GraphTree            ==#
    def GetOrderedLabelListFromGraphTree(self):
        print('The method [GetOrderedLabelListFromGraphTree] is currently not implemented!')
        return None

    #==          Create edge-matrix from GraphTree            ==#
    def GetSemiEncodedDataset(self):
        print('The method [GetSemiEncodedDataset] is currently not implemented!')
        return None

    #===========================================================#
    #==                      Build Methods                    ==#
    #===========================================================#

    #==               Build a tree like graph                 ==#
    # This function build a Graph as tree structure with labeld nodes, 
    # which can usde to navigate like in a graph.
    def BuildTreeLikeGraphFromRLDAG(self, orderedNodesDepth, orderedNodesContent):
        if (isODict(orderedNodesDepth)) and (isODict(orderedNodesContent)):
            if(len(orderedNodesDepth) != len(orderedNodesContent)):
                print('INSERTIONS CONTENT SIZE NOT EQUAL AND ERROR FOR [BuildTreeLikeGraphFromRLDAG]')
                return None
            else:
                root = None
                prev_index = -1
                prev_node = None
                depth = None
                label = None
                content = None

                # For all ordered graphnodes gathered by there depth in the graph
                for index in range(len(orderedNodesDepth)):
                    depth = orderedNodesDepth[index]
                    label, content = self.GetSplittedContent(orderedNodesContent[index])

                    #Setup rooted node
                    if(index == 0):
                        root = self.NewAnyNode( index,
                                                'root',
                                                depth,
                                                False,
                                                [],
                                                True,
                                                [],
                                                label,
                                                content)

                        prev_node = root

                    #Handle the subgraph parts
                    else:
                        if label is '':
                            print('ERROR! => [', orderedNodesContent[index], ']')

                        prev_node = self.BuildNextNode( prev_node,
                                                        index,
                                                        orderedNodesDepth[prev_index],
                                                        None,
                                                        depth,
                                                        True,
                                                        [],
                                                        False,
                                                        [],
                                                        label,
                                                        content)

                    prev_index = index
            return root
        else:
            print('WRONG INPUT FOR [BuildTreeLikeGraphFromRLDAG]')
            return None

    #==                    Create a new nodes                 ==#
    # This function creates a new AnyNode with the given input.
    def NewAnyNode(self, nId, nState, nDepth, nHasInputNode, nInputNode, nHasFollowerNodes, nFollowerNodes, nLabel, nContent):
        if isInt(nId) and isBool(nHasInputNode) and isBool(nHasFollowerNodes) and isList(nFollowerNodes):
            if(nHasInputNode == False):
                    nInputNode = None

            return AnyNode( id=nId,
                            name=nState,
                            state=nState,
                            depth=nDepth,
                            hasInputNode=nHasInputNode,
                            parent=nInputNode,
                            hasFollowerNodes=nHasFollowerNodes,
                            followerNodes=nFollowerNodes,
                            label=nLabel,
                            content=nContent)
        else:
            print('WRONG INPUT FOR [NewAnyNode]')
            return None

    #==                   Build next RLDAG node               ==#
    # This function create a new node depending on a given node or tree.
    # Here the node gets its position inside the tree depending on the given prev node which is the root or a tree structure.
    def BuildNextNode(self, prev_node, index, p_depth, state, depth, hasInputs, input, hasFollowers, followers, label, content):
        if isInt(index) and isInt(depth) and isInt(p_depth):
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
        else:
            print('WRONG INPUT FOR [BuildNextNode]')
            return None


    #===========================================================#
    #==                 Node Manipulation Methods             ==#
    #===========================================================#

    #==             Set Node state if not navigated           ==#
    # This funtion set the state of a node in the GraphTree.
    def NormalState(self, node):
        if isAnyNode(node):
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
        else:
            print('WRONG INPUT FOR [NormalState]')

    #=        Check Node is navigated and set state           ==#
    # This function chech notes if they are root, subnode or child (-> DAG review as Tree)
    # If this check is true it calls NormalState function.
    # Otherwise it sets state to 'navigator'.
    # Graph of Rank 1 is a sigle root node!
    def NavigateState(self, graph_root, node):
        if isAnyNode(graph_root) and isAnyNode(node):
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
                    #print('CONTROL: ', ShowRLDAGTree(graph_root))
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
        else:
            print('WRONG INPUT FOR [NavigateState]')

    #=                  Get direct subnodes                  ==#
    # This function place sub nodes to the current node if some exist.
    def GetSubnodes(self, node):
        if isAnyNode(node):
            followers = [node.label for node in node.children]
            if(len(followers) > 0):
                node.hasFollowerNodes = True
                node.followerNodes = followers
        else:
            print('WRONG INPUT FOR [GetSubnodes]')

    #==                   Reforge single node                ==#
    # This function wrap state setup and child info placement for 1 node.
    def SingleNodeReforge(self, graph_root, node):
        if isAnyNode(graph_root) and isAnyNode(node):
            self.NavigateState(graph_root, node)
            self.GetSubnodes(node)
        else:
            print('WRONG INPUT FOR [SingleNodeReforge]')

    #==                   Reforge all nodes                  ==#
    # This function wrap the SingleNodeReforge for all GraphTree nodes.
    def GraphReforge(self, root):
        if (isAnyNode(root)) and (isNotNone(root.children)):
            for node in PreOrderIter(root):
                self.SingleNodeReforge(root, node)
        else:
            print('WRONG INPUT FOR [ReforgeGraphContent]')

    def ShowGathererInfo(self, amr_stringified, root):
        """
        This function print the input and result of the gatherer.
        This allow to evaluate the gatherers work easily.
            :param self: 
            :param amr_stringified: 
            :param root: 
        """   
        try:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(amr_stringified)
            self.ShowRLDAGTree(root)
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.ShowGathererInfo]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    #==       Preprocessor for AMR-String-Representation      ==#
    # This function fixes format problems and collect nodes depth and there content ordered by appearance.
    # So its possible to rebuild the AMR structure.
    def AMRPreprocessor(self, semantic_flag, graph_nodes, nodes_depth, nodes_content):
        """
        This function fixes format problems and collect nodes depth and there content ordered by appearance.
        So its possible to rebuild the AMR structure.
            :param semantic_flag: 
            :param graph_nodes: 
            :param nodes_depth: 
            :param nodes_content: 
        """   
        try:
            v = 6   # Definition of 1 depth step in AMR
            k = 0
            
            # Each line is a node definition
            for line in graph_nodes:
                if(semantic_flag not in line) and (line != ''):
                    s = len(line) - len(line.lstrip(' '))
                    t_rest = s%v
                    t = s/v

                    if(t_rest > 0):
                        line, s = self.AddLeadingWhitespaces(line, (v-t_rest))
                        t = s/v

                    if(k > 0) and ((t - nodes_depth[k-1]) > 1):
                        nodes_depth[k] = nodes_depth[k-1]+1
                    else:
                        nodes_depth[k] = t

                    nodes_depth[k] = toInt(t)
                    nodes_content[k] = self.CleanNodeSequence(line)

                    if nodes_content[k] is '':
                        print('Raw: ', line)

                    k = k + 1
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.AMRPreprocessor]. Arguments:\n{1!r}"
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
            graph_nodes = amr_stringified.split('\n')
            nodes_depth = OrderedDict()
            nodes_content = OrderedDict()

            self.AMRPreprocessor(semantic_flag, graph_nodes, nodes_depth, nodes_content)
            root = self.BuildTreeLikeGraphFromRLDAG(nodes_depth, nodes_content)
            self.GraphReforge(root)

            if(print_to_console):
                self.ShowGathererInfo(amr_stringified, root)

            return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.Pipeline]. Arguments:\n{1!r}"
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
            if(self.saving):
                return self.ExportToJson(root)
            else:
                return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TreeParser.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)