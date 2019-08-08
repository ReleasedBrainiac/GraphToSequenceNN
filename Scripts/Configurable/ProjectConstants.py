class Constants():
    """
    This class provides necessary constants for the whole Graph2Sequence Tool.
    """
    # Raw Datafile Constants
    ELEMENT_SPLIT_REGEX:str = '\#\s+'

    # Look up Datafile Constants
    MAPPING_SPLIT_REGEX:str = '\#'

    # Dataset Provider Constants
    TYP_ERROR:str = 'Entered wrong type! Input is no String!'
    SIGN_ERROR:str = 'Unauthorized sign found!'
    PARENTHESIS_ERROR:str = 'Unauthorized parenthtesis found!'
    SENTENCE_DELIM:str = '::snt'
    SEMANTIC_DELIM:str = '::smt'
    FILE_DELIM:str = '::file'
    START_SIGN:str = "<GO>"
    END_SIGN:str = "<EOS>"

    # AMR Cleaner Constants
    INDENTATION:int = 6
    COLON:str = ':'
    QUOTATION_MARK:str = '"'
    CONNECTOR:str = '-'
    POLARITY:str = ' - '
    WHITESPACE:str = ' '
    NEG_POLARITY:str = 'not'
    POS_POLITE:str = 'positive'
    NEG_POL_LABEL:str = 'N0T'

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

    NP_GATHER_LOAD_SHAPE_REX:str = r'(# Array shape: )?([0-9, ()]+)'

    NP_TEACHER_FORCING_FILE_NAMES:list = ["_nodes_emb.out", "_forward_look_up.out", "_backward_look_up.out", "_vecs_input_words.out", "_vecs_target_words.out"]
