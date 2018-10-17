# - *- coding: utf-8*-
'''
    Resrources:
        => https://code.visualstudio.com/docs/python/environments#_select-an-environment 
'''


from Scripts.TestHandler.TestCores import EvaluateTests, ReportTestProgress, RunTests
from Scripts.TestHandler.Tests.GraphConverterTesting import GraphTests
'''
    Running Tests!
'''
def RunGraphHandlerTests():
    print('>>>>>>>>>>>> RUNNING TESTS! <<<<<<<<<<<<')
    results = RunTests(GraphTests)
    print('\n')

    print('>>>>>>>>>> EVALUATING TESTS! <<<<<<<<<<<')
    EvaluateTests(results)
    print('\n')
    return None
