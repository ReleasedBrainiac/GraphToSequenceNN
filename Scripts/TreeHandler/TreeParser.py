import re
from anytree import AnyNode, RenderTree, find, findall, PreOrderIter
from collections import OrderedDict
from DatasetHandler.ContentSupport import isODict, isList, isNone, isNotEmptyString
from DatasetHandler.ContentSupport import isInStr, isNotInStr, isNotNone
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from Configurable.ProjectConstants import Constants

class TParser:
    """
    This class library allow to extract content from AMR String representation.
    The result will provide as AnyNode or Json-AnyNode representation.

    Used Resources:
        => http://anytree.readthedocs.io/en/latest/
    """

    def __init__(self, amr_str:str =None, show_process:bool =False, is_saving:bool =False):
        """
        This class constructor collects necessary inputs and initialize the constants only.
            :param amr_str:str: amr input as string
            :param show_process:bool: show processing steps
            :param is_saving:bool: result saving data or further processing
        """   
        try:
            self.constants = Constants()
            self.amr_input = amr_str
            self.show = show_process
            self.saving = is_saving
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExportToJson(self, anytree_root:AnyNode):
        """
        This function converts a AnyNode tree to a AnyNode Json representation containing the initial '#::smt' flag.
            :param anytree_root:AnyNode: root of an anytree object
        """   
        try:
            return '#::smt\n' + JsonExporter(indent=2, sort_keys=True).export(anytree_root) + '\n'
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ExportToJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ImportAsJson(self, json:str):
        """
        This function converts a AnyNode Json representation to a AnyNode tree.
            :param json:str: anytree json 
        """   
        try:
            return JsonImporter().import_(json.replace('#::smt\n', ''))
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ImportAsJson]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ExtractRawNodeSequence(self, sequence:str):
        """
        This function collects the raw node sequence containing the label 
        and existing features like flags and descriptional content.
            :param sequence:str: node sequence
        """   
        try:
            node_sentence = sequence.lstrip(' ')+')'
            return node_sentence[node_sentence.find("(")+1:node_sentence.find(")")] 
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ExtractRawNodeSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectNodeDefinition(self, input_node:str):
        """
        This function collects the full node definition from amr substring variable.
        Depending on the node it will contain at least the label and maybe additional features.
            :param input_node:str: amr substring containing at least just 1 node.
        """   
        try:
            parts = input_node.lstrip(' ').split('/')
            if len(parts) == 1:
                return parts, None
            else:
                return parts[0], parts[1]
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.CollectNodeDefinition]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def RemoveExtensionsAndFlags(self, node_elements:list):
        """
        This function removes all word extensions and flag elements in a given node_sequence
            :param node_elements:list: split elements defining the node sequence
        """   
        try:
            results = []

            if isNotNone(node_elements):
                for node_element in node_elements:
                    if isInStr("-", node_element) or  isInStr(":", node_element):
                        if isInStr("-", node_element) and isNotInStr(":", node_element):
                            sub_sequence = node_element[0: node_element.rfind('-')]
                            if(len(sub_sequence) > 0): results.append(sub_sequence)
                        else:
                            continue
                    else:
                        if(len(node_element) > 0): results.append(node_element)

            return results
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.RemoveExtensionsAndFlags]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CleanNodeSequence(self, sequence:str):
        """
        This function cleans the node sequence and returns label and content only.
        It will also cuts of word extensions so it just get the basis word of a nodes content!
            :param sequence:str: node sequence
        """   
        try:
            node_seq = self.ExtractRawNodeSequence(sequence)
            if(isInStr(' ', node_seq)):
                elements = node_seq.split(' ')
                results = self.RemoveExtensionsAndFlags(elements)
                node_seq = ' '.join(results)

            return node_seq
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.CleanNodeSequence]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ShowRLDAGTree(self, root:AnyNode):
        """
        This function shows a rendered representation of a AnyNode tree on console!
            :param root:AnyNode: root of a anytree
        """   
        try:
            print(RenderTree(root))
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ShowRLDAGTree]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetParentWithPrev(self, node:AnyNode, at_depth:int):
        """
        This function collects the parent node of the given node depending on a given depth.
        The depth is needed to find the correct node during navigation through a graph construction.
        We search with the previous node and define the steps we have to step up in the tree. 
            :param node:AnyNode: current node
            :param at_depth:int: depth of the parent
        """   
        try:
            cur_parent = node.parent
            while((cur_parent.depth + 1) > at_depth):
                cur_parent = cur_parent.parent

            return cur_parent
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetParentWithPrev]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GetParentOfNewNode(self, depth:int, p_depth:int, prev_node:AnyNode):
        """
        This function collects the parent for a new node.
            :param depth:int: depth of the new node
            :param p_depth:int: depth of the previous inserted node in the tree
            :param prev_node:AnyNode: previous node 
        """   
        try:
            if(depth > p_depth):
                return prev_node
            else:
                if(depth == p_depth):
                    return prev_node.parent
                else:
                    return self.GetParentWithPrev(prev_node, depth)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GetParentOfNewNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildTreeGraph(self, nodes_depths:OrderedDict, nodes_contents:OrderedDict):
        """
        This function build a graph like tree from (R)outed (L)abeled (D)irected (A)cyclic (G)raph [RLDAG].
            :param nodes_depths:OrderedDict: list of nodes depths in order of occurrence
            :param nodes_contents:OrderedDict: list of nodes labels and featues in order of occurrence
        """   
        try:
            if(len(nodes_depths) != len(nodes_contents)): return None
            else:
                root = None
                prev_index = -1
                prev_node = None
                depth = None
                label = None
                content = None

                for index in range(len(nodes_depths)):
                    depth = nodes_depths[index]
                    label, content = self.CollectNodeDefinition(nodes_contents[index])

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
                    else:
                        parent_depth = nodes_depths[prev_index]
                        prev_node = self.BuildNextNode( prev_node, index, parent_depth, None, depth, True, [], False, [], label, content)

                    prev_index = index
            return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.BuildTreeGraph]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NewAnyNode(self, n_id:int, n_state:str, n_depth:int, n_has_input:bool, n_inputs:list, n_has_followers:bool, n_followers:list, n_label:str, n_content:str):
        """
        This function creates a new node containing the given params.
            :param n_id:int: node id
            :param n_state:str: node type
            :param n_depth:int: node depth
            :param n_has_input:bool: has at least 1 parent
            :param n_inputs:list: parents
            :param n_has_followers:bool: has at least 1 children
            :param n_followers:list: childrens
            :param n_label:str: node label
            :param n_content:str: features stored in the node
        """   
        try:
            if(n_has_input == False): n_inputs = None

            return AnyNode( id=n_id,
                            name=n_state,
                            state=n_state,
                            depth=n_depth,
                            hasInputNode=n_has_input,
                            parent=n_inputs,
                            hasFollowerNodes=n_has_followers,
                            followerNodes=n_followers,
                            label=n_label,
                            content=n_content)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.NewAnyNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildNextNode(self, prev_node:AnyNode, index:int, p_depth:int, state:str, depth:int, hasInputs:bool, input:list, hasFollowers:bool, followers:list, label:str, content:str):
        """
        This function create a new node depending on a given node or tree.
        Here the node gets its position inside the tree depending on the given prev node which is the root or a tree structure.

            PARAMS @see ~> NewAnyNode 
        """   
        try:
            if(index > 0):
                inputs = self.GetParentOfNewNode(depth, p_depth, prev_node)
                return self.NewAnyNode( n_id=index,
                                        n_state=state,
                                        n_depth=depth,
                                        n_has_input=hasInputs,
                                        n_inputs=inputs,
                                        n_has_followers=hasFollowers,
                                        n_followers=followers,
                                        n_label=label,
                                        n_content=content)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.BuildNextNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def NormalState(self, node:AnyNode):
        """
        This function sets the state of a node in the GraphTree.
            :param node:AnyNode: a node
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

    def NavigateState(self, graph_root:AnyNode, node:AnyNode):
        """
        This function sets the state of a node depending on its (position in the) corresponding tree (-> DAG review as Tree)
            :param graph_root:AnyNode: tree root
            :param node:AnyNode: node from tree you want to update
        """   
        try:
            if isNotNone(node.label) and isNone(node.content):
                label = node.label
                regex = str('\\b'+label+'\\b')
                desired = []

                tmp_desired = findall(graph_root, lambda node: node.label in label)

                for i in tmp_desired:
                    match = re.findall(regex, i.label)
                    if len(match) > 0:
                        desired.append(i)

                if(len(desired) < 1):  print( node.state )
                elif(len(desired) == 1): self.NormalState(node)
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

    def SetSubnodes(self, node:AnyNode):
        """
        This function places sub nodes to the current node if some exist.
            :param node:AnyNode: given node
        """   
        try:
            followers = [node.label for node in node.children]
            if(len(followers) > 0):
                node.hasFollowerNodes = True
                node.followerNodes = followers
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.SetSubnodes]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def GraphReforge(self, root:AnyNode):
        """
        This function wraps the SingleNodeReforge for all GraphTree nodes.
            :param root:AnyNode: tree root
        """
        try:
            for node in PreOrderIter(root):
                self.NavigateState(root, node)
                self.SetSubnodes(node)
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.GraphReforge]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def ShowGathererInfo(self, amr_str:str, root:AnyNode):
        """
        This function prints the input and result of the gatherer.
            :param amr_str:str: given amr
            :param root:AnyNode: given tree
        """   
        try:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(amr_str)
            self.ShowRLDAGTree(root)
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.ShowGathererInfo]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Preprocessor(self, graph_nodes:list, nodes_depth:OrderedDict, nodes_content:OrderedDict):
        """
        This functions fills the depths and content dicts with informations about the amr.
            :param graph_nodes:list: amr splitted by line
            :param nodes_depth:OrderedDict: collecting depth dict
            :param nodes_content:OrderedDict: collecting content dict
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

    def Pipeline(self, amr_str:str, print_to_console:bool):
        """
        This function gather a TreeGraph as AnyNode Tree from AMR-String-Representation.
            :param amr_str:str: single amr definition
            :param print_to_console:bool: show processing steps 
        """   
        try:
            nodes_depth = OrderedDict()
            nodes_content = OrderedDict()
            self.Preprocessor(amr_str.split('\n'), nodes_depth, nodes_content)
            root = self.BuildTreeGraph(nodes_depth, nodes_content)
            self.GraphReforge(root)
            if(print_to_console): self.ShowGathererInfo(amr_str, root)
            return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Execute(self):
        """
        This function handle the execution of the gatherer.
        The result is a importable json if you like to store [is_saving = True] 
        or a AnyNode Tree if you want to process it further [is_saving = False]
        """   
        try:
            root = self.Pipeline(self.amr_input, self.show)
            if(self.saving): return self.ExportToJson(root)
            else: return root
        except Exception as ex:
            template = "An exception of type {0} occurred in [TParser.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)