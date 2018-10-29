from networkx import nx
from Configurable.ProjectConstants import Constants
from DatasetHandler.ContentSupport import isInStr, isNotNone, isStr, isDict, isList

class GParser:

    _constants = None
    _context = None
    _graph_fragments = None
    _paranthesis = ['(',')']
    _look_up_table = None

    def __init__(self, in_context, in_look_up_table, in_parenthesis=['(',')']):
        """
        docstring here
            :param self: 
            :param in_context: 
            :type in_context: str
            :param in_look_up_table:
            :type in_look_up_table: dict
            :param in_parenthesis: 
            :type in_parenthesis: list<str>[2]
        """   
        try:
            self.constants = Constants()
            if(isStr(in_context)): self._context = in_context
            if(isList(in_parenthesis) and (len(in_parenthesis) == 2 )): self._paranthesis = in_parenthesis
            if(isDict(in_look_up_table)): self._look_up_table = in_look_up_table

            print('####################################################')
            print(in_context, '\n ???????????????? \n')
        except Exception as ex:
            template = "An exception of type {0} occurred in [GParser]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
    
    def GetNetworkxGraph(self):
        """
        This method return the NetworkX graph of a passed AMR string.
            :param self: 
        """   
        try:
            return self.Pipeline()
        except Exception as ex:
            template = "An exception of type {0} occurred in [GParser.GetNetworkxGraph]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Pipeline(self):
        try:
            self._graph_fragments = self._context.split('\n')
            self.BuildInitialGraph()

        except Exception as ex:
            template = "An exception of type {0} occurred in [GParser.Pipeline]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def BuildInitialGraph(self):
        try:
            step_size = 6   # Definition of 1 depth step in AMR
            node_index = 0
            
            # Each line is a node definition
            for line in self._graph_fragments:
                if(self.constants.SEMANTIC_DELIM not in line) and (line != ''):
                    s = len(line) - len(line.lstrip(' '))
                    t_rest = s%step_size
                    t = s/step_size

                    if(t_rest > 0):
                        line, s = self.AddLeadingWhitespaces(line, (step_size-t_rest))
                    print(line,' \t => [S: ',s,'| C: ',node_index,'| T: ',(s/step_size),']')


                    '''
                    if(node_index > 0) and ((t - nodes_depth[node_index-1]) > 1):
                        nodes_depth[node_index] = nodes_depth[node_index-1]+1
                    else:
                        nodes_depth[node_index] = t

                    nodes_depth[node_index] = toInt(t)
                    nodes_content[node_index] = self.CleanNodeSequence(line)

                    if nodes_content[node_index] is '':
                        print('Raw: ', line)
                    '''
                    node_index = node_index + 1

            print('#########################################\n')

        except Exception as ex:
            template = "An exception of type {0} occurred in [GParser.BuildInitialGraph]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


    def AddLeadingWhitespaces(self, str, amount):
        try:
            for _ in range(amount): str = ' '+str

            ws_count = len(str) - len(str.lstrip(' '))
            return [str, ws_count]
        except Exception as ex:
            template = "An exception of type {0} occurred in [GParser.AddLeadingWhitespaces]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)