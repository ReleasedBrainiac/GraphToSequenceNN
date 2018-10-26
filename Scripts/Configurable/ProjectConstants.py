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
    CONNECTOR = '-'
    POLARITY = ' - '
    WHITESPACE = ' '
    NEG_POLARITY = 'not'
    NEG_POL_LABEL = 'NT0'

    # Regex
    EXTENSION_REGEX = r'(\w*(\-\w+)+)'
    EXTENSION_NUMBER_REGEX = r'\-\d+'
    EXTENSION_WORD_ENDING_REGEX = r'(\w+\-)+[0-9]+'
    EXTENSION_MULTI_WORD_REGEX = r'[^:](\w*(\-[a-zA-Z]+)+)'
    
    SIGN_POLARITY_REGEX = r'(\(\-\))|(\s+\-[\r\t\f ]*)'
    SIGNS_REMOVE_UNUSED_REGEX = r'[^a-zA-Z\d\s:\/\:\-\(\)]'
    

    MARKER_NESTED_STR_REGEX = r'\B(\"\w+( \w+)*\")'
    FLAG_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'

    ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'
    UNENCLOSED_ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z0-9\-]+)'