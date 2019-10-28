from collections import OrderedDict
from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isNotEmptyString, getIndexedODictLookUp
from DatasetHandler.ContentSupport import MatrixExpansionWithZeros
import numpy as np
import sys

class MatrixBuilder:
    """
    This class provide amr context to graph matrices builder.
    """
    def __init__(self, context:str =None, show_feedback:bool =False):
        """
        This class constructor stores the given context and allow to activation of showing the process results on the console.
            :param context:str: amr input
            :param show_feedback:bool: show process response on console
        """   
        try:
            self.input_semantic = context
            self.show_response = show_feedback
            self.constants = Constants()
            self.graph_nodes = OrderedDict()

            # THIS IS DEPRECATED!
            #self.nodes_as_dict = False
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def MatrixSplitOnPrincipalDiagonal(self, edge_matrix:np.ndarray):
        """
        This function split a whole [M x N] adjacent edge matrix of a graph into 2 [M x N] adjacent matrices.
        The 1. matrix will then contain all forward connections of origin edge matrix by zerofying all values under the principal diagonal. 
        The 2. matrix will contain the incomming connections of each node by transposing the 1. matrix.

        Attention: A directed graph will be threatened as a bidirected graph because we want to use it for text mapping, later. 
                   In this case there is no need to keep concrete graph directions.

        Additional: The algorith could use the numpy.matrix.transpose but it would cost more performance than creating both submatrices at once, so transpose is not used.

            :param edge_matrix:np.ndarray: [M x N] edge matrix containing all forward and backward connections of a graph.
        """
        try:
            forward_connections = np.zeros(edge_matrix.shape)
            backward_connections = np.zeros(edge_matrix.shape)

            r_index = -1
            for row in edge_matrix:
                r_index += 1
                e_index = -1
                for entry in row:
                    e_index += 1
                    if r_index >= e_index:
                        forward_connections[r_index][e_index] = 0
                        backward_connections[r_index][e_index] = entry
                    else:
                        forward_connections[r_index][e_index] = entry
                        backward_connections[r_index][e_index] = 0

            stack = np.stack((forward_connections, backward_connections), axis=0)
            return stack
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.MatrixSplitOnPrincipalDiagonal]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
    def BuildEdgeMatrix(self, connections:list, ordered_vertex:OrderedDict):
        """
        This function build a edge matrix on a given connections list and return it with it's corresponding vertex list.
            :param connections:list: found connections
            :param ordered_vertex:OrderedDict: collected verticies
        """   
        try:
            array_format = (len(ordered_vertex), len(ordered_vertex))
            edge_matrix = np.zeros(array_format)
            indexed_verticies = getIndexedODictLookUp(ordered_vertex)

            for edge in connections:
                start_index = indexed_verticies[edge[0]]
                end_index = indexed_verticies[edge[1]]
                edge_matrix[start_index, end_index] = 1
                edge_matrix[end_index,start_index] = 1

            return [edge_matrix, indexed_verticies]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.BuildEdgeMatrix]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def CollectNewNode(self, node_string:str):
        """
        This function collect amr node definitions from amr node string.
            :param node_string:str: amr string fragment containing a node definition
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
                    if parts[0].isdigit():
                        self.graph_nodes[parts[0]] = parts[0]
                    else:
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
            
            edges, verticies = self.BuildEdgeMatrix(connections=connections_list, ordered_vertex=self.graph_nodes)
            edge_pairs = self.MatrixSplitOnPrincipalDiagonal(edges)

            if self.show_response:
                print('#####################################')
                print('Input:\n', self.input_semantic)
                print('Nodes:\n',self.graph_nodes)
                print('Cons.:\n',connections_list)
                print('Vertices:\n', verticies)
                print('Edges:\n', edge_pairs)

            return [edge_pairs, list(self.graph_nodes.values())]
            
            # THIS IS DEPRECATED!
            #if self.nodes_as_dict:
            #    return [edge_pairs, self.graph_nodes]
            #else:
            #    return [edge_pairs, list(self.graph_nodes.values())]

        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(0)

class MatrixHandler():
    """
    This class provide matrix equalization algorithms to expand and zero fill matrices.
    """
    def SingleLookUpEqualization(self, datapair:list, max_card:int):
        """
        This function wraps the MatrixExpansionWithZeros function for the foward and backward edge look up for a datapair.
            :param datapair:list: single elemt of the DatasetPipeline result 
            :param max_card:int: desired max cardinality 
        """
        try:
            assert (datapair[1][0] is not None), ('Wrong input for dataset edge look up size equalization!')
            elem1 = MatrixExpansionWithZeros(datapair[1][0][0], max_card)
            elem2 = MatrixExpansionWithZeros(datapair[1][0][1], max_card)
            assert (elem1.shape == elem2.shape and elem1.shape == (max_card,max_card)), ("Results have wrong shape!")
            return [elem1,elem2]
        except Exception as ex:
            template = "An exception of type {0} occurred in [MatrixHandler.SingleLookUpEqualization]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

    def DatasetLookUpEqualization(self, datapairs:list ,max_cardinality:int):
        """
        This function equalizes all datasets neighbourhood look up matrices to a given max cardinality.
            :param datapairs:list: the dataset
            :param max_cardinality:int: the given cardinality
        """   
        try:
            assert (max_cardinality > 0), ("Max graph nodes cardinality was 0!")
            for datapair in datapairs:
                datapair[1][0] = self.SingleLookUpEqualization(datapair, max_cardinality)
        except Exception as ex:
            template = "An exception of type {0} occurred in [MatrixHandler.DatasetLookUpEqualization]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)    