from keras import backend as K

def ConsoleReporter(label, printable,evaluate=True):
    AssertIsTensor(printable)
    if evaluate:
        print(label,':\n',KerasEval(printable),'\n')
    else:
        print(label,':\n',printable,'\n')
        
def AssertTensorShapeEqual(tensor_x, tensor_y):
    assert (tensor_x.shape == tensor_y.shape), ('Assertion failed shapes dont match! [',tensor_x.shape,'] dont match " [',tensor_y.shape,']')  
    
def AssertNotNone(tensor, name):
    assert (tensor is not None),('The given input',name,'was None!')
    
def AssertNotNegative(value):
    assert (value >= 0), ('Input value was negative')

def AssertTensorDotDim(tensor_x, tensor_y):
    AssertIsTensor(tensor_x)
    AssertIsTensor(tensor_y)
    x_shape = tensor_x.shape
    y_shape = tensor_y.shape
    
    if x_shape != y_shape:
        x_dim = x_shape[1]
        y_dim = y_shape[0]
        assert (x_dim == y_dim),('Assertion failed for matrix multiplication cause [',x_dim,'] * [',y_dim,'] dont match!')

def AssertAddTensorToTensor(tensor, add_tensor):
    if tensor.shape != add_tensor.shape:
        assert (tensor.shape[1] == add_tensor.shape[0]),("Weighted tensor can't be savely biased. This is caused by shape missmatch!")
    
def AssertIsTensor(tensor_x):
    assert (K.is_tensor(tensor_x)),('Given input is no tensor! =>', tensor_x)

def AssertIsKerasTensor(tensor_x):
    assert (K.is_keras_tensor(tensor_x)),('Given input is no KERAS tensor! =>', tensor_x)

def KerasShape(tensor_x):
    AssertIsTensor(tensor_x)
    return K.shape(tensor_x)

def KerasEval(tensor_x):
    AssertIsTensor(tensor_x)
    return K.eval(tensor_x)