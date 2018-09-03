# - *- coding: utf-8*-
'''
    Resrources:
        => https://code.visualstudio.com/docs/python/environments#_select-an-environment 
'''


from TestHandler.TestCores import EvaluateTests, ReportTestProgress, RunTests
from TestHandler.Tests.GraphConverterTesting import GraphTests
'''
    Running Tests!
'''

print('>>>>>>>>>>>> RUNNING TESTS! <<<<<<<<<<<<')
results = RunTests(GraphTests)
print('\n')

print('>>>>>>>>>> EVALUATING TESTS! <<<<<<<<<<<')
EvaluateTests(results)
print('\n')
