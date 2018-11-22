from collections import OrderedDict
from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isNotEmptyString, getIndexedODictLookUp
import numpy as np

class MatrixBuilder:

    input_semantic = None
    constants = None
    graph_nodes = None

    def __init__(self, input):
        try:
            self.input_semantic = input
            self.constants = Constants()
            self.graph_nodes = OrderedDict()
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildNpEdgeMatrix(self, edge_dict, ordered_vertex_dict):
        try:
            array_format = (len(ordered_vertex_dict), len(ordered_vertex_dict))
            edge_matrix = np.zeros(array_format)

            for edge in edge_dict:
                start_vertex = edge[0]
                end_vertex = edge[1]
                indexed_verticies = getIndexedODictLookUp(ordered_vertex_dict)

                start_index = indexed_verticies[start_vertex]
                end_index = indexed_verticies[end_vertex]

                edge_matrix[start_index, end_index] = 1
                edge_matrix[end_index,start_index] = 1

            return [edge_matrix, indexed_verticies]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.BuildNpEdgeMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectNewNode(self, node_string):
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
            print('#####################################')
            print('Input:\n', self.input_semantic)
            print('Nodes:\n',self.graph_nodes)
            print('Cons.:\n',connections_list)
            print('Vertices:\n', verticies)
            print('Edges:\n', edges)
            return [edges, verticies]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        