# - *- coding: utf-8*-
from TextFormatting.ContentSupport import isNotNone, isList, isBool, isStr
import inspect

def ReportTestProgress(check_value, test_name):
    
    if check_value: 
        return True
    else:
        print(test_name, '[FAILED!]')
        return False

def RunTests(class_name):
    functions = dir(class_name)
    TestCases = []

    for func in functions:
        if '__' not in func:
            func_placeholder = getattr(class_name, func)

            if inspect.isfunction(func_placeholder):
                print('Running tests [',func,']')
                TestCases.append([func_placeholder(), func])

    return TestCases

def EvaluateTests(TestResults):
    ALL_TEST_SUCCEED = True
    ALL_FAILED_TEST = []

    if isNotNone(TestResults) and isList(TestResults):
        for elem, name in TestResults:
            if isNotNone(elem) and isBool(elem) and isNotNone(name) and isStr(name):
                if not elem:
                    ALL_FAILED_TEST.append([elem, name])
                    ALL_TEST_SUCCEED = False
            else:
                print('WRONG CONTENT FOUND IN LIST AT [EvaluateTests]')
                return None

        if ALL_TEST_SUCCEED:
            print('ALL TESTS SUCCEEDED!')
            return True
        else:
            for result, name in ALL_FAILED_TEST:
                print('TEST [',name, '] RETURNED [',result, ']!')
            return False

    else:
        print('WRONG INPUT FOR [EvaluateTests]')
        return None