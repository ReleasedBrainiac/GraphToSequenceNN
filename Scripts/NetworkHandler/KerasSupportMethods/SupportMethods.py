from keras import backend as K

def ConsoleReporter(label, printable,evaluate=True):
    if evaluate and AssertIsTensor(printable):
        print(label,':\n',KerasEval(printable),'\n')
    else:
        print(label,':\n',printable,'\n')
        
def AssertTensorShapeEqual(tensor_x, tensor_y):
    assert (tensor_x.shape == tensor_y.shape), ('Assertion failed shapes dont match! [',tensor_x.shape,'] dont match " [',tensor_y.shape,']')
    
def AssertVectorLength(vector_x, vector_y):
    assert (vector_x.shape[0] == vector_y.shape[0]), ('Vector lenghts dont match! [',vector_x.shape[0],'] dont match " [',vector_y.shape[0],']')    
    
def AssertTensorDotDim(tensor_x, tensor_y):
    AssertIsTensor(tensor_x)
    AssertIsTensor(tensor_y)
    x_dim_0 = tensor_x.shape[0]
    y_dim_1 = tensor_y.shape[1]
    assert (x_dim_0 == y_dim_1),('Assertion failed for matrix multiplication cause [',x_dim_0,'] * [',y_dim_1,'] dont match!')
    
def AssertIsTensor(tensor_x):
    check = K.is_tensor(tensor_x)
    assert (check),('Given input is no tensor! =>', tensor_x)
    return check

def AssertIsKerasTensor(tensor_x):
    check = K.is_keras_tensor(tensor_x)
    assert (check),('Given input is no KERAS tensor! =>', tensor_x)
    return check

def KerasShape(tensor_x):
    AssertIsTensor(tensor_x)
    return K.shape(tensor_x)

def KerasEval(tensor_x):
    AssertIsTensor(tensor_x)
    return K.eval(tensor_x)