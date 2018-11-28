from collections import OrderedDict
from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isNotEmptyString, getIndexedODictLookUp
import numpy as np

class MatrixBuilder:

    input_semantic = None
    constants = None
    graph_nodes = None
    show_response = False

    def __init__(self, context, show_console_reponse=False):
        """
        This class constructor stores the given context and allow to activation of showing the process results on the console.
            :param context: amr input string
            :param show_console_reponse: switch allows to show process response on console or not
        """   
        try:
            self.input_semantic = context
            self.show_response = show_console_reponse
            self.constants = Constants()
            self.graph_nodes = OrderedDict()
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildNpEdgeMatrix(self, connections_dict, ordered_vertex_dict):
        """
        This function build a edge matrix on a given connections list and return it with its corresponding vertex list.
            :param connections_dict: dict of found connections
            :param ordered_vertex_dict: ordered of collected verticies
        """   
        try:
            array_format = (len(ordered_vertex_dict), len(ordered_vertex_dict))
            edge_matrix = np.zeros(array_format)
            indexed_verticies = getIndexedODictLookUp(ordered_vertex_dict)

            for edge in connections_dict:
                start_index = indexed_verticies[edge[0]]
                end_index = indexed_verticies[edge[1]]
                edge_matrix[start_index, end_index] = 1
                edge_matrix[end_index,start_index] = 1

            return [edge_matrix, indexed_verticies]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.BuildNpEdgeMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectNewNode(self, node_string):
        """
        This function collect amr node definitions from amr node string.
            :param node_string: amr string fragment containing a node definition
        """   
        try:
            parts = node_string.replace(' ', '').replace('\n', '').replace('(', '').replace(')', '').split('/')
            if len(parts) > 1:
                if parts[0] not in self.graph_nodes:
                    self.graph_nodes[parts[0]] = parts[1].replace(' ', '')
                else:
                    if (self.graph_nodes[parts[0]] == None) and (parts[1] != None):
                        self.graph_nodes[parts[0]] = parts[1].replace(' ', '')
            else: 
                if parts[0] not in self.graph_nodes:
                    self.graph_nodes[parts[0]] = None

            return parts[0]    
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.CollectNewNode]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Execute(self):
        """
        This function execute the matrix building and vertex collecting process.
        """   
        try:
            nodes_stack = []
            connections_list = []
            lines = self.input_semantic.split('(')
            self.graph_nodes = OrderedDict()

            for line in lines:
                if (self.constants.SEMANTIC_DELIM not in line) and (isNotEmptyString(line)):
                    prev_node = None
                    next_node = self.CollectNewNode(line)
                    
                    if len(nodes_stack) < 1:
                        nodes_stack.append(next_node)
                    else:
                        closer = line.count(')')
                        if closer < 2:
                            prev_node = nodes_stack.pop()
                            nodes_stack.append(prev_node)
                            if closer == 0:
                                nodes_stack.append(next_node) 
                        else:
                            prev_node = nodes_stack.pop()
                            for _ in range(closer-2):  
                                nodes_stack.pop()

                    if prev_node != None: connections_list.append([prev_node, next_node])
            
            edges, verticies = self.BuildNpEdgeMatrix(connections_list, self.graph_nodes)

            if self.show_response:
                print('#####################################')
                print('Input:\n', self.input_semantic)
                print('Nodes:\n',self.graph_nodes)
                print('Cons.:\n',connections_list)
                print('Vertices:\n', verticies)
                print('Edges:\n', edges)
            
            return [edges, self.graph_nodes]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        