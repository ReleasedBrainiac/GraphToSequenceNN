# - *- coding: utf- 8*-
# Source to build the tree like graph structure is anytree
# ~> http://anytree.readthedocs.io/en/latest/
import re
from anytree import AnyNode, RenderTree, find, findall, PreOrderIter
from collections import OrderedDict
from TextFormatting.Contentsupport import isDict, multiIsDict
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter

#===========================================================#
#==                      Helper Methods                   ==#
#===========================================================#

#==               Get nodes label and content             ==#
def GetSplittedContent(str):
    if('/' not in str):
        return str, None
    else:
        parts = str.split('/')
        label = parts[0].replace(" ", "")
        content = parts[1].replace(" ", "")
        return label, content

#==                    Cleanup AMR spacing                ==#
def AddLeadingWhitespaces(str, amount):
    for i in range(amount):
        str = ' '+str

    ws_count = len(str) - len(str.lstrip(' '))
    return [str, ws_count]

#==                 Get max range of a graph              ==#
def GetMaxRange(nodes):
    if(isDict(nodes)):
        max_depth = 0
        for node in nodes:
            if(max_depth < nodes.get(node)):
                max_depth = nodes.get(node)
        return int(max_depth)+1 #add+1 during range beginns on 0
    else:
        print('WRONG INPUT TYPE FOR [GetMaxRange]')
        return None

#==          Cleaning nodes sequence from garbage         ==#
def CleanNodeSequence(node_sequence):
    node_sent = re.sub(' +',' ',node_sequence+')')
    node_sent = node_sent[node_sent.find("(")+1:node_sent.find(")")]

    if(' ' in node_sent):
        elements = node_sent.split(' ')
        results = []
        for value in elements:
            if("-" in value) or (":" in value):
                if("-" in value) and not (":" in value):
                    str1 = value[0: value.rfind('-')]
                    if(len(str1) > 0):
                        results.append(str1)
                else:
                    continue
            else:
                if(len(value) > 0):
                    results.append(value)
        node_sent = ' '.join(results)
    else:
        node_sent = node_sent[0: node_sent.rfind('-')]

    return node_sent

#===========================================================#
#==                    Gatherer methods                   ==#
#===========================================================#

def NodeOrderOnDepth(nodes):
    order = OrderedDict()
    max_depth = GetMaxRange(nodes)

    for value in range(max_depth):
        layer = OrderedDict()
        for node in nodes:
            current_node = dict()

            if(nodes.get(node) == value):
                current_node['position'] = node
                current_node['state'] = None
                current_node['depth'] = value
                layer[node] = current_node

        order[value] = layer

    print('Deepest: ', max_depth)

    for value in order:
        print('Layers: ', order[value], '\n\n')

#===========================================================#
#==               Build a tree like graph                 ==#
#===========================================================#

def BuildTreeLikeGraphFromRLDAG(orderedNodesDepth, orderedNodesContent):
    if(len(orderedNodesDepth) != len(orderedNodesContent)):
        print('Inserted List dont match in size FOR [BuildTreeLikeGraphFromDAG]')
        return None
    else:
        root = None
        raw_cnt = None
        prev_index = -1

        prev_node = None
        depth = None
        label = None
        content = None

        for index in range(len(orderedNodesDepth)):
            raw_cnt = orderedNodesContent[index]
            depth = orderedNodesDepth[index]

            label, content = GetSplittedContent(raw_cnt)

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
                                   orderedNodesDepth[index],
                                   orderedNodesDepth[prev_index],
                                   None,
                                   depth,
                                   True,
                                   [],
                                   None,
                                   [],
                                   label,
                                   content)

            prev_index = index
    return root

#==             Show RLDAG tree representation            ==#
def ShowRLDAGTree(root):
    print(RenderTree(root))

#==                   Build next RLDAG node               ==#
def BuildNextNode( prev_node,
                   index,
                   c_depth,
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
        if(c_depth > p_depth):
            input = prev_node
        else:
            #Next same layer Node in the RLDAG
            if(c_depth == p_depth):
                input = prev_node.parent
            #Next rising layer Node in the RLDAG
            else:
                input = GetParent(prev_node, c_depth)

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

#==               Search for depending parent             ==#
def GetParent(in_node, cur_depth):
    cur_parent = in_node.parent
    if(cur_parent != None) and ((cur_parent.depth + 1) > cur_depth):
        while((cur_parent.depth + 1) > cur_depth):
            cur_parent = cur_parent.parent

    return cur_parent

#==                    Create a new nodes                 ==#
def NewAnyNode(nId, nState, nDepth, nHasInputNode, nInputNode, nHasFollowerNodes, nFollowerNodes, nLabel, nContent):
    if(nHasInputNode == False):
        return AnyNode( id=nId,
                        name=nState,
                        state=nState,
                        depth=nDepth,
                        hasInputNode=nHasInputNode,
                        parent=None,
                        hasFollowerNodes=nHasFollowerNodes,
                        followerNodes=nFollowerNodes,
                        label=nLabel,
                        content=nContent)
    else:
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

#==              Set destination, root or else           ==#
def NodeSetNormalState(node):
    if(isinstance(node, AnyNode)):
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

#=                  Get direct subnodes                  ==#
def GetSubnodes(node):
    if(isinstance(node, AnyNode)):
        followers = [node.label for node in node.children]
        if(len(followers) > 0):
            node.hasFollowerNodes = True
            node.followerNodes = followers
    else:
        print('WRONG INPUT FOR [GetSubnodes]')

#==                   Reforge single node                ==#
def SingleNodeReforge(graph_root, node):
    if isinstance(graph_root, AnyNode) and isinstance(node, AnyNode):
        # Get node state and connected subnodes
        StateDefinition(graph_root, node)
        GetSubnodes(node)
    else:
        print('WRONG INPUT FOR [SingleNodeReforge]')

#==                   Connect informations                ==#
def ReforgeGraphContent(graph_root):
    if(isinstance(graph_root, AnyNode)):
        if(graph_root.children is not None):
            #nodes = [node for node in PreOrderIter(graph_root)]
            for node in PreOrderIter(graph_root):
                SingleNodeReforge(graph_root, node);
    else:
        print('WRONG INPUT FOR [ReforgeGraphContent]')
        return None

#==                    Export informations                ==#
def ExportToJson(root):
    exporter = JsonExporter(indent=2, sort_keys=True)
    return '#::smt\n' + exporter.export(root) + '\n'

#==                    Import informations                ==#
def ImportAsJson(json_string):
    importer = JsonImporter()
    return importer.import_(data)

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

            nodes_depth[k] = t
            nodes_content[k] = CleanNodeSequence(line)
            k = k + 1

    root = BuildTreeLikeGraphFromRLDAG(nodes_depth, nodes_content)
    #ShowRLDAGTree(root)
    ReforgeGraphContent(root)
    ShowRLDAGTree(root)
    return ExportToJson(root)
    
