class Constants():
    """
    This class provides necessary constants for the whole Graph2Sequence Tool.
    """
    # Raw Datafile Constants
    ELEMENT_SPLIT_REGEX = '\#\s+'

    # Look up Datafile Constants
    MAPPING_SPLIT_REGEX = '\#'

    # Dataset Provider Constants
    TYP_ERROR = 'Entered wrong type! Input is no String!'
    SIGN_ERROR = 'Unauthorized sign found!'
    PARENTHESIS_ERROR = 'Unauthorized parenthtesis found!'
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
    POS_POLITE = 'positive'
    NEG_POL_LABEL = 'N0T'

    # Regex
    EXTENSION_REGEX = r'(\w*(\-\w+)+)'
    EXTENSION_NUMBER_REGEX = r'\-\d+'
    EXTENSION_WORD_ENDING_REGEX = r'(\w+\-)+[0-9]+'
    EXTENSION_MULTI_WORD_REGEX = r'[^:](\w*(\-[a-zA-Z]+)+)'
    
    SIGN_POLITE_REGEX = r'(\(\[\+\]\))|(\s+\+[\r\t\f ]*)|(\(\+\))'

    SIGN_POLARITY_REGEX = r'(\(\[\-\]\))|(\s+\-[\r\t\f ]*)|(\(\-\))'
    SIGNS_REMOVE_UNUSED_REGEX = r'[^a-zA-Z\d\s\/\-\(\)\[\]\?]'

    NUMBER_QUANTITIY_REGEX = r'(\(\[[0-9]+\]\))'

    MARKER_NESTINGS_REGEX = r'(\"(.*?)\")'
    FLAG_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( \d)*'

    ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z]+)'
    UNENCLOSED_ARGS_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z0-9+\-]+)'
    UNENCLOSED_ARGS_MULTIWORD_REGEX = r'\B\:(?=\S*[+-]*)([a-zA-Z0-9*-]+)+( +[a-zA-Z0-9+\-_]+)'
    MISSING_CAPTURED_NEG_REGEX = r'\B( - )'