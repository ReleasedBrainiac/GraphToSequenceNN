from keras import backend as K

def ConsoleReporter(label, printable, evaluate:bool =True):
    """
    This function print a tensor as converted console feedback.
        :param label: prefix label
        :param printable: printable object
        :param evaluate:bool: print evaluated keras tensor result
    """
    try:   
        AssertIsTensor(printable)
        if evaluate:
                print(label,':\n',KerasEval(printable),'\n')
        else:
                print(label,':\n',printable,'\n')
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.ConsoleReporter]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        
def AssertTensorShapeEqual(tensor_x, tensor_y):
    """
    This assert checks 2 tensors being equal in there shape definition.
        :param tensor_x: 1st tensor
        :param tensor_y: 2nd tensor
    """
    try:
        assert (tensor_x.shape == tensor_y.shape), ('Assertion failed shapes dont match! [',tensor_x.shape,'] dont match " [',tensor_y.shape,']')  
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertTensorShapeEqual]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    
def AssertNotNone(tensor, name):
    """
    This function evaluates a tensor isn't None.
        :param tensor: given tensor
        :param name: name of the tensor
    """
    try:
        if name is None: name = ''
        assert (tensor is not None),('The given input',name,'was None!')
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertNotNone]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    
def AssertNotNegative(value:int):
    """
        This function checks a integer issn't lower then 0.
        :param value:int: given integer
    """  
    try:
        assert (value >= 0), ('Input value was negative')
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertNotNegative]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def AssertTensorDotDim(tensor_x, tensor_y):
    """
    This function checks 2 tensor are being able to perform a Keras.dot operation.
        :param tensor_x: 1st tensor
        :param tensor_y: 2nd tensor
    """   
    try:
        AssertIsTensor(tensor_x)
        AssertIsTensor(tensor_y)
        x_shape = tensor_x.shape
        y_shape = tensor_y.shape
        
        if x_shape != y_shape:
                x_dim = x_shape[1]
                y_dim = y_shape[0]
                assert (x_dim == y_dim),('Assertion failed for matrix multiplication cause [',x_dim,'] * [',y_dim,'] dont match!')
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertTensorDotDim]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def AssertAddTensorToTensor(tensor, add_tensor):
    """
    This fucntions asserts 2 can perform an Add function. Parameter order will be treated as Add function input order.
    This function is used mostly for bias add control.
        :param tensor: 1st tensor
        :param add_tensor: 2nd tensor
    """   
    try:
        if tensor.shape != add_tensor.shape:
                assert (tensor.shape[1] == add_tensor.shape[0]),("Can't add tensors savely. This is caused by shape missmatch!")
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertAddTensorToTensor]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    
def AssertIsTensor(tensor_x):
    """
    This function asserts a given input is a tensor.
        :param tensor_x: possible tensor input
    """
    try:
        assert (IsTensor(tensor_x, None)),('Given input is no tensor! =>', tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertIsTensor]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

def AssertIsKerasTensor(tensor_x):
    """
    This function asserts a given input is a keras tensor.
        :param tensor_x: possible keras tensor input
    """
    try:
        assert (IsKerasTensor(tensor_x, None)),('Given input is no KERAS tensor! =>', tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.AssertIsKerasTensor]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message) 

def KerasShape(tensor_x):
    """
    This function returns the shape of a tensor. Its a wraper of the K.shape function with additional control.
        :param tensor_x: given tensor
    """
    try:
        AssertIsTensor(tensor_x)
        return K.shape(tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.KerasShape]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message) 

def KerasEval(tensor_x):
    """
    This functions evaluates the value insider a keras tensor.
        :param tensor_x: given keras tensor
    """
    try:
        AssertIsTensor(tensor_x)
        AssertIsKerasTensor(tensor_x)
        return K.eval(tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.KerasEval]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message) 

def IsTensor(tensor_x, name):
    """
    This function is a wrapper for K.is_tensor with additional assertion.
        :param tensor_x: tensor 
        :param name: tensor name
    """   
    try:
        AssertNotNone(tensor_x, name)
        return K.is_tensor(tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.IsTensor]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message) 

def IsKerasTensor(tensor_x, name):
    """
    This function is a wrapper for K.is_keras_tensor with additional assertion.
        :param tensor_x: keras tensor 
        :param name: tensor name
    """
    try:
        AssertNotNone(tensor_x, name)
        return K.is_keras_tensor(tensor_x)
    except Exception as ex:
        template = "An exception of type {0} occurred in [SupportMethods.IsKerasTensor]. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message) 