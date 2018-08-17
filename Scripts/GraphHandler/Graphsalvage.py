# - *- coding: utf- 8*-
# Source to build the tree like graph structure is anytree
# ~> http://anytree.readthedocs.io/en/latest/
import re
from anytree import AnyNode, RenderTree, find, findall, PreOrderIter
from collections import OrderedDict
from TextFormatting.Contentsupport import *
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter

#===========================================================#
#==                    Conversion Methods                 ==#
#===========================================================#

#==                    Export informations                ==#
# This function allow to convert a AnyNode Tree to a AnyNode-JsonString representation.
def ExportToJson(root):
    if(isAnyNode(root)):
        exporter = JsonExporter(indent=2, sort_keys=True)
        return '#::smt\n' + exporter.export(root) + '\n'
    else:
        print('WRONG INPUT FOR [ExportToJson]')
        return None

#==                    Import informations                ==#
# This function allow to convert a AnyNode-JsonString representation to a AnyNode Tree.
def ImportAsJson(json):
    if(isStr(json)):
        importer = JsonImporter()
        return importer.import_(json)
    else:
        print('WRONG INPUT FOR [ImportAsJson]')
        return None

#===========================================================#
#==                      Helper Methods                   ==#
#===========================================================#

#==         Get nodes sentence from string sequence       ==#
# This function allow to collect the raw node sequence containing the label 
# and maybe some additional values like flags and description content.
def ExtractRawNodeSequence(sequence):
    if(isStr(sequence)):
        node_sent = re.sub(' +',' ',sequence+')')
        return node_sent[node_sent.find("(")+1:node_sent.find(")")]
    else:
        print('WRONG INPUT FOR [ExtractSentence]')
        return None

#==               Get nodes label and content             ==#
# This function allow to collect the Label and eventually a label content 
# from string of a ARM Graph variable.
def GetSplittedContent(str):
    if(isStr(str)):
        if(isNotInStr('/', str)):
            return str, None
        else:
            parts = str.split('/')
            label = parts[0].replace(" ", "")
            content = parts[1].replace(" ", "")
            return label, content
    else:
        print('WRONG INPUT FOR [GetSplittedContent]')
        return None

#==                    Cleanup AMR spacing                ==#
# This function clean up the whitespacing in the ARM Graphstring by a given value.
def AddLeadingWhitespaces(str, amount):
    if(isStr(str)) and (isInt(amount)):
        for i in range(amount):
            str = ' '+str

        ws_count = len(str) - len(str.lstrip(' '))
        return [str, ws_count]

    else:
        print('WRONG INPUT FOR [AddLeadingWhitespaces]')
        return None

#==          Cleaning nodes sequence from garbage         ==#
# This function allow to clean the node sequence from all staff so it return label and content only.
# It will also cut of word extensions so we just get the basis word of a nodes content!
def CleanNodeSequence(sequence):
    if (isStr(sequence)):
        node_seq = ExtractRawNodeSequence(sequence)

        # If we have more then just a label
        if(isInStr(' ', node_seq)):
            elements = node_seq.split(' ')
            results = []
        
            # Clean the content
            for value in elements:
                if(isInStr("-", value)) or (isInStr(":", value)):
                    if(isInStr("-", value)) and (isNotInStr(":", value)):
                        str1 = value[0: value.rfind('-')]
                        if(len(str1) > 0):
                            # Control Structure
                            #print("[Raw: ", node_seq, "| Value: ",value,"| Clean: ",str1,"]")
                            results.append(str1)
                    else:
                        continue
                else:
                    if(len(value) > 0):
                        results.append(value)
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
def ShowRLDAGTree(root):
    if(isAnyNode(root)):
        print(RenderTree(root))
    else:
        print('WRONG INPUT FOR [ShowRLDAGTree]')

#==               Search for depending parent             ==#
# This function allow to get the parent node of the given node depending on a given depth.
# The depth is needed to find the correct node during navigation through a graph construction.
# We search with the previous node and define the steps we have to step up in the tree. 
def GetParentWithPrev(node, cur_depth):
    if isAnyNode(node) and isInt(cur_depth) and isNotNone(cur_depth):
        cur_parent = node.parent
        while((cur_parent.depth + 1) > cur_depth):
            cur_parent = cur_parent.parent

        return cur_parent
    else:
        print('WRONG INPUT FOR [GetParentWithPrev]')
        return None

#===========================================================#
#==                      Build Methods                   ==#
#===========================================================#

#==               Build a tree like graph                 ==#
# This function build a Graph as tree structure with labeld nodes, 
# which can usde to navigate like in a graph.
def BuildTreeLikeGraphFromRLDAG(orderedNodesDepth, orderedNodesContent):
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
                label, content = GetSplittedContent(orderedNodesContent[index])

                #Setup rooted node
                if(index == 0):
                    root = NewAnyNode( index,
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
                    prev_node = BuildNextNode( prev_node,
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


#==                   Build next RLDAG node               ==#
def BuildNextNode( prev_node,
                   index,
                   p_depth,
                   state,
                   depth,
                   hasInputs,
                   input,
                   hasFollowers,
                   followers,
                   label,
                   content):

    #Handle the subgraph parts
    if(index > 0):
        #Next step down in the Rooted Label DAG (RLDAG)
        if(depth > p_depth):
            input = prev_node
        else:
            #Next same layer Node in the RLDAG
            if(depth == p_depth):
                input = prev_node.parent
            #Next rising layer Node in the RLDAG
            else:
                input = GetParentWithPrev(prev_node, depth)

        prev_node = NewAnyNode( index,
                    state,
                    depth,
                    hasInputs,
                    input,
                    hasFollowers,
                    followers,
                    label,
                    content)

    return prev_node

#==                    Create a new nodes                 ==#
def NewAnyNode(nId, nState, nDepth, nHasInputNode, nInputNode, nHasFollowerNodes, nFollowerNodes, nLabel, nContent):

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

#===========================================================#
#==                 Node Manipulation Methods             ==#
#===========================================================#

#==              Set destination, root or else           ==#
def NodeSetNormalState(node):
    if isAnyNode(node):
        if(node.is_leaf):
            node.state = 'destination'
        elif(node.is_root):
            node.state = 'root'
            node.parent = None
        else:
            node.state = 'subnode'
    else:
        print('WRONG INPUT FOR [NodeSetNormalState]')

#=              Check node is navigated                  ==#
def StateDefinition(graph_root, node):
    if isAnyNode(graph_root) and isAnyNode(node):
        if(node.label != None) and (node.content == None):
            label = node.label
            desired =findall(graph_root, lambda node: node.label in label)
            if(len(desired) == 1):
                NodeSetNormalState(node)
            else:
                node.followerNodes = desired[0].followerNodes
                node.hasFollowerNodes = desired[0].hasFollowerNodes
                node.hasInputNode = desired[0].hasInputNode
                node.state = ('navigator')
        else:
            NodeSetNormalState(node)
    else:
        print('WRONG INPUT FOR [StateDefinition]')

#=                  Get direct subnodes                  ==#
def GetSubnodes(node):
    if isAnyNode(node):
        followers = [node.label for node in node.children]
        if(len(followers) > 0):
            node.hasFollowerNodes = True
            node.followerNodes = followers
    else:
        print('WRONG INPUT FOR [GetSubnodes]')

#==                   Reforge single node                ==#
def SingleNodeReforge(graph_root, node):
    if isAnyNode(graph_root) and isAnyNode(node):
        # Get node state and connected subnodes
        StateDefinition(graph_root, node)
        GetSubnodes(node)
    else:
        print('WRONG INPUT FOR [SingleNodeReforge]')
        return None

#==                   Connect informations                ==#
def ReforgeGraphContent(root):
    if (isAnyNode(root)) and (isNotNone(root.children)):
        for node in PreOrderIter(root):
            SingleNodeReforge(root, node);
    else:
        print('WRONG INPUT FOR [ReforgeGraphContent]')

#===========================================================#
#==             Gather the  graph informations            ==#
#==     We create ordered by input order dictionairies    ==#
#===========================================================#

def GatherGraphInfo(graph, sem_flag):
    v = 6
    k = 0
    root = None
    graphLines = graph.split('\n')
    nodes_depth = OrderedDict()
    nodes_content = OrderedDict()
    #print(graph)
    
    for line in graphLines:
        if(sem_flag not in line) and (line != ''):
            s = len(line) - len(line.lstrip(' '))
            t_rest = s%v
            t = s/v

            if(t_rest > 0):
                line, s = AddLeadingWhitespaces(line, (v-t_rest))
                t = s/v

            if(k > 0) and ((t - nodes_depth[k-1]) > 1):
                nodes_depth[k] = nodes_depth[k-1]+1
            else:
                nodes_depth[k] = t

            nodes_depth[k] = toInt(t)
            nodes_content[k] = CleanNodeSequence(line)
            k = k + 1

    root = BuildTreeLikeGraphFromRLDAG(nodes_depth, nodes_content)
    #ShowRLDAGTree(root)
    ReforgeGraphContent(root)
    ShowRLDAGTree(root)
    return ExportToJson(root)
    
