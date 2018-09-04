from TestHandler.TestCores import ReportTestProgress as RTP
from TextFormatting.NodeStringExtractor import CountSubsStrInStr, CheckOpenEnclosing, GetEnclosedContent
from TextFormatting.ContentSupport import isNone

OPEN_PAR = '('
CLOSING_PAR = ')'

NONE_CONTENT = 'None Content'
NONE_OPEN = 'None Open'
NONE_CLOSE = 'None Close'
IS_ERROR = 'Result of Error'
IS_VALID = 'Result of Valid'

class NodeStringExtractorTest:

    def TestInputCountSubs():
        expected_value = 4
        search = OPEN_PAR
        error = '((()'
        valid = '(((()))'

        # Test cases for the methods with report to console
        react_none_str = RTP( 0 == CountSubsStrInStr(None, search), NONE_CONTENT)
        
        react_none_search = RTP(0 == CountSubsStrInStr(valid, None), 'NoneSearchString')
        
        react_invalid_str = RTP((expected_value != CountSubsStrInStr(error, search)), IS_ERROR)
        
        react_valid_str = RTP((expected_value == CountSubsStrInStr(valid, search)), IS_VALID)

        # United result
        return (react_none_str      and 
                react_none_search   and 
                react_invalid_str   and 
                react_valid_str)

    def TestCheckOpenEnclosing():
        error = '(((fadsf,)aa)'
        valid = '(ya)(i,(sdaf)44s(sfs))'

        # Test cases for the methods with report to console
        react_none_content = RTP(isNone(CheckOpenEnclosing(None, OPEN_PAR, CLOSING_PAR)), NONE_CONTENT)

        react_none_open = RTP(isNone(CheckOpenEnclosing(valid, None, CLOSING_PAR)), NONE_OPEN)

        react_none_close = RTP(isNone(CheckOpenEnclosing(valid, OPEN_PAR, None)), NONE_CLOSE)

        react_fail_str = RTP(False == CheckOpenEnclosing(error, OPEN_PAR, CLOSING_PAR), IS_ERROR)

        react_valid_str = RTP(CheckOpenEnclosing(valid, OPEN_PAR, CLOSING_PAR), IS_VALID)

        # United result
        return (react_none_content  and 
                react_none_open     and 
                react_none_close    and 
                react_fail_str      and 
                react_valid_str)

    def TestGetEnclosedContent():
        error = '[fail!)'
        valid = '(success!)'
        inside_valid = 'success!'

        # Test cases for the methods with report to console
        react_none_content = RTP(isNone(GetEnclosedContent(None, OPEN_PAR, CLOSING_PAR)), NONE_CONTENT)

        react_none_open = RTP(isNone(GetEnclosedContent(valid, None, CLOSING_PAR)), NONE_OPEN)

        react_none_close = RTP(isNone(GetEnclosedContent(valid, OPEN_PAR, None)), NONE_CLOSE)

        react_fail_str = RTP(isNone(GetEnclosedContent(error, OPEN_PAR, CLOSING_PAR)), IS_ERROR)

        react_valid_str = RTP(inside_valid == GetEnclosedContent(valid, OPEN_PAR, CLOSING_PAR), IS_VALID)

        # United result
        return (react_none_content  and 
                react_none_open     and 
                react_none_close    and 
                react_fail_str      and 
                react_valid_str)

    def Tets():