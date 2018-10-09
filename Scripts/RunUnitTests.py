# - *- coding: utf-8*-
# Linter issue => https://github.com/DonJayamanne/pythonVSCode/issues/394
from TestHandler.UnitTestingGraphHandler import RunGraphHandlerTests
from TestHandler.UnitTestingAMRHandler import RunNodeExtractorTests

print('#### RunNodeExtractorTests ####')
RunNodeExtractorTests()
print()

print('#### RunGraphHandlerTests ####')
RunGraphHandlerTests()