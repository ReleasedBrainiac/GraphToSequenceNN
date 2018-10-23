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

    # Regex
    EXTENSION_REGEX = r'(\w*(\-\w+)+)'
    EXTENSION_ELEMENT_REGEX = r'\-\d+'
    FIND_EXTENSION_HAZRDS = r'\s*(((\()*(\w+ )(\/ \w+)*\)*)*|(\(\w+\)))'
    
    OLD_POLARITY_SIGN_REGEX = r'\s+\-\s*'
    POLARITY_SIGN_REGEX = r'(\(\-\))|(\s+\-[\r\t\f ]*)'

    QUALIFIED_STR_REGEX = r'\B(\"\w+( \w+)*\")'
    FLAG_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'

    ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'
    UNENCLOSED_ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z0-9\-]+)'