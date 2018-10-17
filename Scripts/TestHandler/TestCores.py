# - *- coding: utf-8*-
from Scripts.TextFormatting.ContentSupport import isNotNone, isList, isBool, isStr
import inspect

def ReportTestProgress(check_value, test_name):
    """
    This function report a console info if the passed test failed.
        :param check_value: result of the test
        :param test_name: name of the test
    """
    if check_value: 
        return True
    else:
        print(test_name, '[FAILED!]')
        return False

def RunTests(class_name):
    """
    This function execute all tests in a test file by calling the class.
        :param class_name: name of class.
    """
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
    """
    This function evaluates all processed tests by list full of booleans representing the test reults.
        :param TestResults: list of booleans representing the test reults
    """
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