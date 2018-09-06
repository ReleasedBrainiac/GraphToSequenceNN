# - *- coding: utf-8*-
'''
    Used Resources:
        => https://www.geeksforgeeks.org/type-isinstance-python/
        => https://anytree.readthedocs.io/en/latest/
        => https://pypi.org/project/ordereddict/#description
        => https://pymotw.com/2/collections/ordereddict.html
        => https://docs.python.org/3.6/library/numbers.html
'''
from anytree import AnyNode
from collections import OrderedDict
from numbers import Number, Real, Rational, Complex
import random as rnd

#####################
# Check and Support #
#####################

def isComplex(input):
    """
    This function check input is a complex number value.
        :param input: unknown type object
    """
    return isinstance(input, Complex)

def isRational(input):
    """
    This function check input is a rational number value.
        :param input: unknown type object
    """
    return isinstance(input, Rational)

def isReal(input):
    """
    This function check input is a real number value.
        :param input: unknown type object
    """
    return isinstance(input, Real)

def isNumber(input):
    """
    This function check input is a number value.
        :param input: unknown type object
    """
    return isinstance(input, Number)

def isBool(input):
    """
    This function check input is a boolean value.
        :param input: unknown type object
    """
    return isinstance(input, bool)

def isNotNone(input):
    """
    This function check input is not None.
        :param input: unknown type object
    """
    return (input is not None)

def isNone(input):
    """
    This function check input is None.
        :param input: unknown type object
    """
    return (input is None)

def isAnyNode(input):
    """
    This function check input is a AnyNode object.
        :param input: unknown type object
    """
    return isinstance(input, AnyNode)

def isList(input):
    """
    This function check input is a list object.
        :param input: unknown type object
    """
    return isinstance(input, list)

def isStr(input):
    """
    This function check input is a string value.
        :param input: unknown type object
    """
    return isinstance(input, str)

def isInt(input):
    """
    This function check input is a integer value.
        :param input: unknown type object
    """
    return isinstance(input, int)
    
def isFloat(input):
    """
    This function check input is a float value.
        :param input: unknown type object
    """
    return isinstance(input, float)

def isODict(input):
    """
    This function check input is a ordered dict object.
        :param input: unknown type object
    """
    return isinstance(input, OrderedDict)

def isDict(input):
    """
    This function check input is a dict object.
        :param input: unknown type object
    """
    return isinstance(input, dict)

def isSet(input):
    """
    This function check input is a set object.
        :param input: unknown type object
    """
    return isinstance(input, set)
    
def hasContent(input):
    """
    This function check sequences/collections containing at least min 1 value.
        :param input: unknown type object
    """
    if isNotNone(input) and isIterable(input):
        return (len(input) > 0)
    else:
        print('WRONG INPUT FOR [hasContent]')
        return False

def isIterable(input):
    """
    This function check a input is iterable,
    for example sequences and collections.
        :param input: unknown type object
    """
    try:
        input_iterator = iter(input)
        return True
    except TypeError as te:
        return False

def getType(input):
    """
    This funtion return the type input
        :param input: unknown type object
    """
    return type(input)

# This function return a int between min and max
# If min and max no integer it return an integer between 0 and 100
def GetRandomInt(min, max):
    if isInt(min) and isInt(max) and (min < max):
        return rnd.randint(min, max)
    else:
        return rnd.randint(0,100)

# This function a string dos not contain a subsequence.
# A subsequence could be a char, string, or a value!
def isNotInStr(seq , string):
    if(isStr(string)) and (isNotNone(seq)):
        return (seq not in string)
    else:
        print('WRONG INPUT FOR [isNotInStr]')
        return False
    

# This function check a string contains a subsequence.
# A subsequence could be a char, string, or a value!
def isInStr(search , content):
    if(isStr(content)) and (isNotNone(search)):
        return (search in content)
    else:
        print('WRONG INPUT FOR [isInStr]')
        return False

# This function check a dictionary contain a specified key.
def singleHasKey(key, input):
    if((isDict(input)) or (isSet(input))) and (isNotNone(key)):
        return(key in input)
    else:
        print('WRONG INPUT FOR [singleHasKey]')
        return False

# This function check a input is a list where each dictionary contain a spezified key.
def multiHasKey(key, input):
    if(isList(input)) and (isNotNone(key)):
        for element in input:
            if(not singleHasKey(key, element)):
                return False

        return True
    else:
        print('WRONG INPUT FOR [multiHasKey]')
        return False
        
# This function check a input is a list of dictionaries.
def multiIsDict(inputs):
    if(isList(inputs)):
        for input in inputs:
            if(not isDict(input)):
                return False
            
        return True
    else:
        print('WRONG INPUT FOR [multiIsDict]')
        return False

# This function allow to set an input or default by a switch value.
def setOrDefault(input, default, wantSet):
    if(isXTypeEqualYType(input, default)) and (isBool(wantSet)):
        if(wantSet):
            return input
        else:
            return default
    else:
        print('WRONG INPUT FOR [setOrDefault]')
        return input

# This function converts a float or integer to an integer value.
def toInt(input):
    if(isFloat(input)) or (isInt(input)):
        return int(input)
    else:
        print('WRONG INPUT FOR [toInt]')
        return None

# This function check input type x and input type y are equal.
def isXTypeEqualYType(in_x, in_y):
    if(isNotNone(in_x)) and (isNotNone(in_y)):
        return (type(in_x) == type(in_y))
    else:
        print('WRONG INPUT FOR [isXTypeEqualYType]')
        return None

# This function return the size of a files content.
def getFileLength(fname):
    if isStr(fname):
        with open(fname, 'r', encoding="utf8") as f:
            for i in enumerate(f):
                pass
        return i + 1
    else:
        print('WRONG INPUT FOR [getFileLength]')
        return 0