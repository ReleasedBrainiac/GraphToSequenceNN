# - *- coding: utf-8*-
'''
    Used Resources:
        => https://www.geeksforgeeks.org/type-isinstance-python/
        => https://anytree.readthedocs.io/en/latest/
        => https://pypi.org/project/ordereddict/#description
        => https://pymotw.com/2/collections/ordereddict.html
        => https://docs.python.org/3.6/library/numbers.html
'''
import random as rnd
from anytree import AnyNode
from collections import OrderedDict
from numbers import Number, Real, Rational, Complex

#####################
# Check and Support #
#####################

# This function check input is a complex number value.
def isComplex(input):
    return isinstance(input, Complex)

# This function check input is a rational number value.
def isRational(input):
    return isinstance(input, Rational)

# This function check input is a real number value.
def isReal(input):
    return isinstance(input, Real)

# This function check input is a number value.
def isNumber(input):
    return isinstance(input, Number)

# This function check input is a boolean value.
def isBool(input):
    return isinstance(input, bool)

# This function check input is not None.
def isNotNone(input):
    return (input is not None)

# This function check input is None.
def isNone(input):
    return (input is None)

# This function check input is a AnyNode object.
def isAnyNode(input):
    return isinstance(input, AnyNode)

# This function check input is a list object.
def isList(input):
    return isinstance(input, list)

# This function check input is a string value.
def isStr(input):
    return isinstance(input, str)

# This function check input is a integer value.
def isInt(input):
    return isinstance(input, int)
    
# This function check input is a float value.
def isFloat(input):
    return isinstance(input, float)

# This function check input is a ordered dict object.
def isODict(input):
    return isinstance(input, OrderedDict)

# This function check input is a dict object.
def isDict(input):
    return isinstance(input, dict)

# This function check input is a set object.
def isSet(input):
    return isinstance(input, set)
    
# This function check sequences/collections containing at least min 1 value.
def hasContent(input):
    if isNotNone(input) and isIterable(input):
        return (len(input) > 0)
    else:
        print('WRONG INPUT FOR [hasContent]')
        return False

# This function check a input is iterable 
# => only possible for sequences and collections.
def isIterable(input):
    try:
        input_iterator = iter(input)
        return True
    except TypeError as te:
        return False

# This funtion return the type input
def GetType(input):
    return type(input)

# This function return a int between min and max
# If min and max no integer it return an integer between 0 and 100
def GetRandomInt(min, max):
    if isInt(min) and isInt(max):
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
def file_len(fname):
        with open(fname, 'r', encoding="utf8") as f:
            for i in enumerate(f):
                pass
        return i + 1