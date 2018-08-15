# - *- coding: utf- 8*-
def file_len(fname):
        with open(fname, 'r', encoding="utf8") as f:
            for i in enumerate(f):
                pass
        return i + 1

def isList(input):
    return isinstance(input, list)

def isStr(input):
    return isinstance(input, str)

def isInt(input):
    return isinstance(input, int)

def isDict(input):
    return isinstance(input, dict)

def isSet(input):
    return isinstance(input, set)

def singleHasKey(key, input):
    if(isDict(input)) or (isSet(input)):
        return(key in input)
    else:
        print('WRONG INPUT TYPE! Neither set nor dict at [singleHasKey]!')
        return False

def multiHasKey(key, input):
    if(isList(input)):
        for element in input:
            if(not singleHasKey(key, element)):
                return False

        return True
    else:
        print('Input is no list at [multiHasKey]!')
        return False
        

def multiIsDict(inputs):
    if(isList(inputs)):
        for input in inputs:
            if(not isDict(input)):
                return False
            
        return True
    else:
        print('Input is no list at [multiIsDict]!')
        return False

def setOrDefault(input, default, setInput):
    if(setInput):
        return input
    else:
        return default
