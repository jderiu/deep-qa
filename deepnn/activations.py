from theano import tensor as T


def relu(x, alpha=0):
    """ 
    Rectified Linear Unit
    """
    return T.switch(x > 0, x, alpha * x)

def relu_f(z):
    """ 
    Wrapper to quickly change the rectified linear unit function
    """
    return z * (z > 0.)

def softplus(x):
    """
    SoftPlus
    """
    return T.nnet.softplus(x)

def sigmoid(x):
    """
    Sigmoid
    """
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    """
    Hard_sigmoid
    """
    return T.nnet.hard_sigmoid(x)

def tanh(x):
    """
    Hyperbolic Tangent
    """
    return T.tanh(x)

def hard_tanh(x):
    """
    The hard version of the Hyperbolic Tangent

    -1 if x < -1
    x if -1 < x < 1
    1 if x > 1 
    """
    return T.minimum(T.maximum(x, -1), 1)

def linear(x):
    '''
    Basically the identity function.
    '''
    return x

def softmax(x):
    """
    a softmax layer
    """
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)