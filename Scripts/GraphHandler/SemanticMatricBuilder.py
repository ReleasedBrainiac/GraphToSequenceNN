from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isNotEmptyString

class MatrixBuilder:

    input_semantic = None
    constants = Constants()

    def __init__(self, input):
        try:
            self.input_semantic = input
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Execute(self):
        try:
            depth = -1
            lines = self.input_semantic.split('\n')

            

            for line in lines:
                if self.constants.SEMANTIC_DELIM not in line:
                    print(line.lstrip(' '))
                    depth = depth + line.count('(')

                    depth = depth - line.count(')')
        except Exception as ex:
            template = "An exception of type {0} occurred in [SemanticMatricBuilder.Execute]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        