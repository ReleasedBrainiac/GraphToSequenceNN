# - *- coding: utf-8*-
'''
    Resrources:
        => https://code.visualstudio.com/docs/python/environments#_select-an-environment 
'''


from Scripts.TestHandler.TestCores import EvaluateTests, ReportTestProgress, RunTests
from Scripts.TestHandler.Tests.AMRConversionTesting import NodeStringExtractorTest
'''
    Running Tests!
'''
def RunNodeExtractorTests():
    print('>>>>>>>>>>>> RUNNING TESTS! <<<<<<<<<<<<')
    results = RunTests(NodeStringExtractorTest)
    print('\n')

    print('>>>>>>>>>> EVALUATING TESTS! <<<<<<<<<<<')
    EvaluateTests(results)
    print('\n')
    return None