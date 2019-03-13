'''
This function library provides data typ controlling statments.

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
import numpy as np
import types

# Basic data type checks must return bool
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

def isIterable(input):
    """
    This function check a input is iterable,
    for example sequences and collections.
        :param input: unknown type object
    """
    try:
        _ = iter(input)
        return True
    except TypeError:
        return False

def isLambda(input):
    """
    This function check input is a lambda type.
        :param input: unknown type object
    """
    return isinstance(input, types.LambdaType)


# Content control statements must return bool
def isXTypeEqualY(object_x, object_y):
    """
    This function check input type x and input type y are equal.
        :param object_x: unknown type object
        :param object_y: unknown type object
    """
    if(isNotNone(object_x)) and (isNotNone(object_y)):
        return (type(object_x) == type(object_y))
    else:
        print('WRONG INPUT FOR [isXTypeEqualY]')
        return None

def isNotEmptyString(input:str):
    """
    this function check a string is not empty.
        :param input:str: given string
    """
    tmp = input.lstrip(' ')
    return isStr(input) and (len(tmp) > 0)

def isNotInStr(search , content):
    """
    This function a string dos not contain a subsequence.
    A subsequence could be a char, string, or a value!
        :param search: string search element 
        :param content: string content element
    """
    if isNotNone(search) and isStr(content):
        if (len(content) >= len(search)):
            return (search not in content)
        else:
            return False
    else:
        print('WRONG INPUT FOR [isNotInStr]')
        return False

def isInStr(search , content):
    """
    This function check a string contains a subsequence.
    A subsequence could be a char, string, or a value!
        :param search: string search element
        :param content: string content element
    """
    if isNotNone(search) and isStr(content):
        if (len(content) >= len(search)):
            return (search in content)
        else:
            return False
    else:
        print('WRONG INPUT FOR [isInStr]')
        return False

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

def singleHasKey(key, input):
    """
    This function check a dictionary contain a specified key.
        :param key: unknown type object
        :param input: a set OR dict
    """
    if(isNotNone(key) and (isDict(input) or isSet(input))):
        return(key in input)
    else:
        print('WRONG INPUT FOR [singleHasKey]')
        return False

def multiHasKey(key, input):
    """
    This function check a input is a list where each dictionary contain a spezified key.
        :param key: unknown type object
        :param input: a list of dictionaries
    """
    if(isList(input)) and (isNotNone(key)):
        for element in input:
            if(not singleHasKey(key, element)):
                return False

        return True
    else:
        print('WRONG INPUT FOR [multiHasKey]')
        return False
        
def multiIsDict(inputs):
    """
    This function check a input is a list of dictionaries.
        :param inputs: a list of dictionaries
    """
    if(isList(inputs)):
        for input in inputs:
            if(not isDict(input)):
                return False
            
        return True
    else:
        print('WRONG INPUT FOR [multiIsDict]')
        return False


# Conversion return casted type value or None
def toInt(input):
    """
    This function converts a float, integer or a number string to an integer value.
        :param input: a string with only numbers OR int OR float 
    """
    if( isFloat(input) or isInt(input) or (isStr(input) and input.isdigit()) ):
        return int(input)
    else:
        print('WRONG INPUT FOR [toInt]')
        return None


# Getters 
def getType(input):
    """
    This funtion return the type input
        :param input: unknown type object
    """
    return type(input)

def getFileLength(path:str):
    """
    This function returns the size of a files content.
        :param path:str: path string
    """
    try:
        with open(path, 'r', encoding="utf8") as f:
            for i in enumerate(f):
                pass
        return i + 1
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.getFileLength]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return 0
        
def getIndexedODictLookUp(dictionary:OrderedDict):
    """
    This getter returns a OrderedDict's key value look up.
        :param dictionary:OrderedDict: given ordered dictionairy
    """
    try:
        ind= {k:i for i,k in enumerate(dictionary.keys())}
        return ind
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.getIndexedODictLookUp]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


# Setters
def setOrDefault(input, default, wantSet:bool):
    """
    This function allow to set an input or default by a switch value.
        :param input: old value of unknown type object
        :param default: default value unknown type object
        :param wantSet:bool: desired value unknown type object
    """
    if isXTypeEqualY(input, default):
        if(wantSet):
            return input
        else:
            return default
    else:
        print('WRONG INPUT FOR [setOrDefault]')
        return input


# Basic list and matrix operations.
def ReorderListByIndices(reorder_list:list, ordering_indices:list):
    """
    This function reorder a list by a given list of ordering indices.
        :param reorder_list:list: list you want to reorder
        :param ordering_indices:list: list of indices with desired value order
    """
    try:
        return [y for x,y in sorted(zip(ordering_indices,reorder_list))] 
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.ReorderListByIndices]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def MatrixExpansionWithZeros(np_2D_array:np.ndarray, up_to_dim:int):
    """
    This function allow to expand a 2D matrix in both directions and fill the empty space with zeros.
    Attention:
    This is only allowed on matrices where the input matrix dimensions are equal and less_equal than the extension value.
        :param np_2D_array:np.ndarray: an numpy array
        :param up_to_dim:int: the dim extension value you desired for x=y
    """
    try:
        AssertEquality(np_2D_array.shape[0], np_2D_array.shape[1], msg="The input matrix dimension aren't equal")
        assert (np_2D_array.shape[0] <= up_to_dim), ("The dimension value isn't less or equal then the arrays dimensions")
        difference = up_to_dim - np_2D_array.shape[0]
        assert (difference > -1), ('Difference was negative!')
        result = np.lib.pad(np_2D_array, (0,difference),'constant', constant_values=(0))
        assert (result.shape[0] == result.shape[1] and result.shape[0] == up_to_dim), ("Result has wrong dimensions!")
        return result
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.MatrixExpansionWithZeros]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def GetRandomInt(min:int, max:int):
    """
    This function return a int between min and max
    If min and max no integer it return an integer between 0 and 100
        :param min:int: minimum 
        :param max:int: maximum
    """
    if (min < max):
        return rnd.randint(min, max)
    else:
        print('WRONG INPUT FOR [GetRandomInt] so range [0,100] was used for generation!')
        return rnd.randint(0,100)

def CalculateMeanValue(str_lengths:list):
    """
    This function calculates the mean over all values in a list.
        :param str_lengths:list: lengths of all strings
    """
    try:
        return int(round(sum(str_lengths)/len(str_lengths)))
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.CalculateMeanValue]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

# Asserts for cases where an exception is necessary.
def AssertNotNone(value, msg:str = ''):
    """
    This assertion alerts on None values.
        :param value: given object
        :param msg:str: [optional] given msg
    """
    warning = msg if (msg != '') else "Given value was None!"
    assert (value is not None), warning

def AssertEquality(first_object, second_object, msg:str = ''):
    """
    This assertion alerts on unequality.
        :param first_object: first given object
        :param second_object: second given object
        :param msg:str: [optional] given msg
    """
    warning = msg if (msg != '') else "Given objects weren't equal!"
    assert (first_object == second_object), warning