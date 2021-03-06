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

def isNdarray(input):
    """
    This function check the input is a numpy ndarray type.
        :param input. unknown type object
    """
    return isinstance(input, np.ndarray)


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

def isNotNegativeNum(input):
    """
    Check a number being positive
        :param input: any type that suffice isNumber
    """
    return (isNumber(input) and abs(input) is input)

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


# Basic values, list and matrix operations.
def CreateNListWithRepeatingValue(repeatable_value, times:int):
    """
    This function allow to generate a list n times long containing at each entry the given repat value!
        :param repeatable_value: value to repeat
        :param times:int: lenght of the list
    """
    try:
        AssertNotNone(repeatable_value, msg="The given repeat value is None! No list will be created!")
        AssertNotNegative(times)
        return [repeatable_value] * times
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.CreateNListWithRepeatingValue]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def SplitBorderAdjustment(dataset_sz:int, split_border:int, desired_batch_sz:int):
    """
    This method allow to fix split border errors to secure batched process execution.
        :param dataset_sz:int: dataset size
        :param split_border:int: calculated solit border index
        :param desired_batch_sz:int: desired datset batch size
    """
    try:
        left_rest:int = split_border%desired_batch_sz
        right_rest:int = (dataset_sz - split_border) % desired_batch_sz
        reducer:int = left_rest + right_rest

        if (reducer >= desired_batch_sz):
            split_border += (desired_batch_sz - left_rest)
            reducer = (desired_batch_sz - right_rest)
            print("1")
        else:
            split_border -= left_rest
            print("2")


        left_rest = split_border%desired_batch_sz
        dataset_right = dataset_sz - split_border - reducer
        right_rest = dataset_right % desired_batch_sz

        if(left_rest == 0) and (right_rest == 0): 
            return split_border, reducer
        else:
            print("Error: Adjustment of the split border value failed!")
            return split_border, None
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.SplitBorderAdjustment]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

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

def MultiDimNdArrayToListedNdArrays(array:np.ndarray):
    try:
        if len(array.shape) > 2:
            return [elem for elem in array]
        else: 
            return array.tolist()
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.MultiDimNdArrayToListedNdArrays]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def RepeatNTimsNdArray(times:int, array:np.ndarray, axis:int = 0):
    """
    This function repeats a given numpy.ndarray desired times and return a 
        :param times:int: repeat value
        :param array:np.ndarray: numpy.ndarray
        :param axis:int: axis to repeat
    """
    try:
        return np.repeat(array[None,:],times, axis=axis)
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.RepeatNTimsNdArray]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def ConcatenateNdArrays(arrays:list, axis:int = 0):
    """
    This function repeats a given numpy.ndarray desired times and return a 
        :param arrays:list: list of np.ndarrays
        :param axis:int: axis to repeat
    """
    try:
        return np.concatenate(arrays, axis=axis)
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.ConcatenateNdArrays]. Arguments:\n{1!r}"
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

def RoundUpRestricted(in_value:int, given_dim:int =100, isBidrectional:bool =True):
    """
    This function return the smallest value that satisfies the rule [in_value % (given_dim * dimension_multiplier) = 0]
        :param in_value:int: given value
        :param given_dim:int=100: desired step size
        :param isBidrectional:bool=True: sets the dimension_multiplier to  1 or 2
    """
    try:
        dimension_multiplier = 2 if isBidrectional else 1
        allowed_dim_products = given_dim * dimension_multiplier
        
        if((in_value%allowed_dim_products) != 0):        
            while(in_value > allowed_dim_products): allowed_dim_products += allowed_dim_products
            return allowed_dim_products
        else:
            return in_value
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.RoundUpRestricted]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def ValueReport(var_name:str, value):
    try:
        print("{}: \n{}\n".format(var_name, value))
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.ValueReport]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def StatusReport(run_index:int, max_index:int, steps:int):
    """
    This function allow to easy provide a verbose report for iteration based steps.
        :param run_index:int: current run index
        :param max_index:int: max run index
        :param steps:int: report steps
    """
    try:
        AssertNotNegative(run_index)
        AssertNotNegative(max_index)
        AssertNotNegative(steps)
        assert (run_index <= max_index), "max index was lower then run index!"
        assert (steps <= max_index), "max index was lower then steps!"

        if (run_index == 0): 
            print("Start Report!")
        if ((run_index+1)% steps == 0 ): 
            print((run_index+1)," / ", max_index)
        if ((run_index+1) == max_index): 
            print("End Report!")
    except Exception as ex:
        template = "An exception of type {0} occurred in [ContentSupport.StatusReport]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def DatasetSplitIndex(dataset_size:int, split_percentage:float):
        """
        This method return a number for the size of desired test samples from dataset by a given percentage.
            :param dataset_size:int: size of the whole datset
            :param split_percentage:float: desired test size percentage from dataset
        """   
        try:
            if isNumber(dataset_size) and isNumber(split_percentage):
                return round(dataset_size * split_percentage)
            else:
                return -1
        except Exception as ex:
            template = "An exception of type {0} occurred in [ContentSupport.DatasetSplitIndex]. Arguments:\n{1!r}"
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

def AssertNotNegative(number, msg:str = ''):
    """
    This assertion alerts on negatives numbers.
        :param value: given object
        :param msg:str: [optional] given msg
    """
    warning = msg if (msg != '') else "Given value was Negative!"
    assert isNotNegativeNum(number), warning

def AssertEquality(first_object, second_object, msg:str = ''):
    """
    This assertion alerts on unequality.
        :param first_object: first given object
        :param second_object: second given object
        :param msg:str: [optional] given msg
    """
    warning = msg if (msg != '') else "Given objects weren't equal!"
    assert (first_object == second_object), warning