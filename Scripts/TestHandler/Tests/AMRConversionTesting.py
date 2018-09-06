# - *- coding: utf-8*-
from TestHandler.TestCores import ReportTestProgress as RTP
from GraphHandler.NodeStringExtractor import CountSubsStrInStr, CheckOpenEnclosing, GetEnclosedContent
from GraphHandler.NodeStringExtractor import DeleteFlags, GenerateCleanAMR
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

    def TestFullExtractor():

        example_1 = """(c / cause-01
      :ARG1 (l / live-01 :polarity -
            :ARG0 (i / i
                  :ARG0-of (t3 / talk-01 :polarity -
                        :ARG2 (a5 / anyone)
                        :ARG1-of (r / real-04)))
            :ARG1 (l2 / life
                  :poss i)
            :manner (a / alone)
            :duration (u / until
                  :op1 (h / have-06
                        :ARG0 i
                        :ARG1 (a3 / accident
                              :mod (p / plane))
                        :location (d / desert :wiki "Sahara" :name (n / name :op1 "Desert" :op2 "of" :op3 "Sahara"))
                        :time (b / before
                              :op1 (n2 / now)
                              :quant (t2 / temporal-quantity :quant 6
                                    :unit (y / year)))))))"""

        result_1 = """(c / cause  
      (l / live
            (NT0 / not)
            (i / i  
                  (t3 / talk
                        (NT0)
                        (a5 / anyone)  
                        (r / real)))  
            (l2 / life 
                  (i))  
            (a / alone)  
            (u / until  
                  (h / have 
                        (i)  
                        (a3 / accident  
                              (p / plane))  
                        (d / desert  
                              (Y0Z / Sahara)  
                              (n / name  
                                    (Y1Z / Desert)  
                                    (Y2Z / of)  
                                    (Y0Z)))  
                        (b / before  
                              (n2 / now)  
                              (t2 / temporal-quantity   
                                    (y / year)))))))"""


        example_2 = """(c / chapter :mod 1)"""
        result_2 = """(c / chapter )"""
        example_3 = """(y2 / yes)"""


        react_none_content  = RTP(isNone(GenerateCleanAMR(None, OPEN_PAR, CLOSING_PAR)), NONE_CONTENT + '_NONE_CONTENT')

        react_none_open     = RTP(isNone(GenerateCleanAMR(example_3, None, CLOSING_PAR)), NONE_CONTENT + '_NONE_OPEN')

        react_none_close    = RTP(isNone(GenerateCleanAMR(example_3, OPEN_PAR, None)), NONE_CONTENT + '_NONE_CLOSE')

        react_no_flag       = RTP(example_3 == GenerateCleanAMR(example_3, OPEN_PAR, CLOSING_PAR), IS_ERROR + '_NO_FLAG')

        react_single_flag   = RTP(result_2 == GenerateCleanAMR(example_2, OPEN_PAR, CLOSING_PAR), IS_VALID + '_SINGLE_FLAG')

        react_multi         = RTP(result_1 == GenerateCleanAMR(example_1, OPEN_PAR, CLOSING_PAR), IS_VALID + '_MULTI')

        # Test cases for the methods with report to console
        # United result
        return (react_none_content   and 
                react_none_open      and 
                react_none_close     and 
                react_no_flag        and 
                react_single_flag    and
                react_multi)