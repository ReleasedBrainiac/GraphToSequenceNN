class Constants:

    # Dataset Provider Constants
    TYP_ERROR = 'Entered wrong type! Input is no String!'
    SENTENCE_DELIM = '::snt'
    SEMANTIC_DELIM = '::smt'
    FILE_DELIM = '::file'

    # AMR Cleaner Constants
    INDENTATION = 6
    COLON = ':'
    QUOTATION_MARK = '"'
    NEG_POLARITY = 'not'
    NEG_POL_LABEL = 'NT0'

    EXTENSION_REGEX = r'\-\d+'
    POLARITY_SIGN_REGEX = r'\s+\-\s*'
    QUALIFIED_STR_REGEX = r'\B(\"\w+( \w+)*\")'
    FLAG_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'
    ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'
    UNENCLOSED_ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z0-9\-]+)'
    #//TODO This regex didnt allow multiple whitespaces in the code!
    FIND_EXTENSION_HAZRDS = r'\s*(((\()*(\w+ )(\/ \w+)*\)*)*|(\(\w+\)))'