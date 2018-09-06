# - *- coding: utf-8*-
from TestHandler.TestCores import ReportTestProgress as RTP
from GraphHandler.NodeStringExtractor import CountSubsStrInStr, CheckOpenEnclosing, GetEnclosedContent
from GraphHandler.NodeStringExtractor import DeleteFlags
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

        react_fail_str = RTP(error == GetEnclosedContent(error, OPEN_PAR, CLOSING_PAR), IS_ERROR)

        react_valid_str = RTP(inside_valid == GetEnclosedContent(valid, OPEN_PAR, CLOSING_PAR), IS_VALID)

        # United result
        return (react_none_content  and 
                react_none_open     and 
                react_none_close    and 
                react_fail_str      and 
                react_valid_str)

    def TestDeleteFlags():
        str_no_flag = 'a / ahead'

        str_with_flag_and_minus = r'a / ahead :ARG1-of'
        str_with_f_and_m_result = r'a / ahead '

        str_only_flag           = r'k / cool :op1'
        str_only_flag_result    = r'k / cool '

        str_multi_flag          = r'n / name :op1 "Desert" :op2 "of" :op3 "Sahara"'
        str_multi_flag_result   = r'n / name  "Desert"  "of"  "Sahara"'
        
        react_none_content      = RTP(isNone(DeleteFlags(None)), NONE_CONTENT)

        react_no_flag           = RTP(str_no_flag == DeleteFlags(str_no_flag), IS_ERROR + '_NO_FLAG')

        react_only_flag         = RTP(str_only_flag_result == DeleteFlags(str_only_flag), IS_VALID + '_ONE_FLAG')

        react_flag_minus        = RTP(str_with_f_and_m_result == DeleteFlags(str_with_flag_and_minus), IS_VALID + '_WITH_ADD_FLAG')

        react_multi_flag_minus  = RTP(str_multi_flag_result == DeleteFlags(str_multi_flag), IS_VALID + '_MULTI_FLAG')

        # Test cases for the methods with report to console
        # United result
        return (react_none_content  and 
                react_no_flag       and 
                react_only_flag     and 
                react_flag_minus    and 
                react_multi_flag_minus)