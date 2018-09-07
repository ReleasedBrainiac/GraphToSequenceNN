# - *- coding: utf-8*-
from anytree import Node, AnyNode, RenderTree
from TreeHandler.TreeParser import NewAnyNode
from TextFormatting.ContentSupport import isNone
from GraphHandler.GraphTreeConverter import GetTensorMatricesFromGraphTree
from TestHandler.TestCores import ReportTestProgress

class GraphTests:
    def NoneTree():
        root = None
        result = ReportTestProgress(isNone(GetTensorMatricesFromGraphTree(root)), 'NoneTree')
        return result

    def NoAnyNode():
        root = Node('root')
        result = ReportTestProgress(isNone(GetTensorMatricesFromGraphTree(root)), 'NoAnyNode')
        return result

    def SingleCorrectResult():
        shape_single = (1,)
        #                ( id,  state,      depth, hasIn,  ins,   hasOuts, outs,    label,  content)
        root = NewAnyNode(  0, 'root',          0, False,   [],     False,  [],     'l',    'lol')

        matrix = GetTensorMatricesFromGraphTree(root)
        result = ReportTestProgress((matrix.shape == shape_single), 'SingleCorrectResult')
        return result

    def MultiCorrectResult():

        #                ( id,  state,      depth, hasIn,  ins,   hasOuts, outs,    label,  content)
        root = NewAnyNode(  0, 'root',          0, False,   [],     True,   [d1,s1], 'h',   'hi')
        d1 = NewAnyNode(    1, 'destination',   1, True,    root,   False,  [],      'i',   'iam')
        s1 = NewAnyNode(    2, 'subnode',       1, True,    root,   True,   [d2],    'j',   'joli')
        d2 = NewAnyNode(    3, 'destination',   2, True,    s1,     False,  [],      'k',   'king')
        print(RenderTree(root))

        #result = True
        result = ReportTestProgress(result, 'MultiCorrectResult')
        return result

    def RedirectCorrectResult():

        #                ( id,  state,      depth, hasIn,  ins,   hasOuts, outs,    label,  content)
        root = NewAnyNode(  0, 'root',          0, False,   [],     True,  [s1,s2], 'h',    'hi')
        s1 = NewAnyNode(    1, 'subnode',       1, True,    root,   True,  [d1],    'i',    'iam')
        d1 = NewAnyNode(    2, 'navigator',     2, True,    s1,     False, [],      'j',    'None')
        s2 = NewAnyNode(    3, 'subnode',       1, True,    root,   True,  [d2],    'j',    'joli')
        d2 = NewAnyNode(    4, 'destination',   2, True,    s2,     False, [],      'k',    'king')

        #result = True
        result = ReportTestProgress(result, 'RedirectCorrectResult')
        return result

    def RedirectWithFollowerNoResult():
        #                ( id,  state,      depth, hasIn,  ins,   hasOuts, outs,    label,  content)
        root = NewAnyNode(  0, 'root',          0, False,   [],     True,  [s1,s2], 'h',    'hi')
        s1 = NewAnyNode(    1, 'subnode',       1, True,    root,   True,  [d1],    'i',    'iam')
        d1 = NewAnyNode(    2, 'navigator',     2, True,    s1,     True,  [d3],    'j',    'None')
        d3 = NewAnyNode(    3, 'destination',   3, True,    d1,     False, [],      'j',    'None')
        s2 = NewAnyNode(    4, 'subnode',       1, True,    root,   True,  [d2],    'j',    'joli')
        d2 = NewAnyNode(    5, 'destination',   2, True,    s2,     False, [],      'k',    'king')

        #result = True
        result = ReportTestProgress(result, 'RedirectCorrectResult')
        return result

