# - *- coding: utf-8*-
'''
    Used Resources:
        => https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.empty.html
'''

from TextFormatting.ContentSupport import GetType
from TextFormatting.ContentSupport import isAnyNode, isNone, isNotNone, isList

'''
    This class library allow to convert GraphTree's of type AnyNode from anytree library into a matrix representation.
'''

def GetNodesLables(tree_graph):
    if isNotNone(tree_graph) and isAnyNode(tree_graph):
        print('This function is work in progress! [GetGloVeFromNodesLables]')
        return None
    else:
        print('WRONG INPUT FOR [GetGloVeFromNodesLables]')
        return None

def GetNeighbourhoodMatrices(tree_graph):
    if isNotNone(tree_graph) and isAnyNode(tree_graph):
        print('This function is work in progress! [GetNeighbourhoodMatrices]')
        return None
    else:
        print('WRONG INPUT FOR [GetNeighbourhoodMatrices]')
        return None

def GetTensorMatricesFromGraphTree(tree_graph):
    if isNotNone(tree_graph) and isAnyNode(tree_graph):
        print('This function is work in progress! [GetTensorMatricesFromGraphTree]')
        return None
    else:
        print('WRONG INPUT FOR [GetTensorMatricesFromGraphTree]')
        return None

def GetDataSet(data_pairs):
    if isNotNone(data_pairs) and isList(data_pairs):
        data_pair_edge_lists = []
        data_pair_features = []

        for _ in range(len(data_pairs)):
            sentence = data_pairs[0][0]
            semantic = data_pairs[0][1]
            edge_lists = GetTensorMatricesFromGraphTree(semantic)

            data_pair_edge_lists.append(edge_lists)
            data_pair_features.append(sentence)

            print('General typ: ',GetType(data_pairs))
            print('Sentence typ: ',GetType(sentence))
            print('Semantic typ: ',GetType(semantic))

        print(data_pair_edge_lists[0])
        print('This function is work in progress! [GetDataSet]')
        return None
    else:
        print('WRONG INPUT FOR [GetDataSet]')
        return None


    