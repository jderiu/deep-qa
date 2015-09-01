import numpy
import theano

def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)

def load_nnet(nnet_fname, params_fname=None):
  print "Loading Nnet topology from", nnet_fname
  train_nnet, test_nnet = cPickle.load(open(nnet_fname, 'rb'))
  if params_fname:
    print "Loading network params from", params_fname
    params = train_nnet.params
    best_params = cPickle.load(open(params_fname, 'rb'))
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)

  for p_train, p_test in zip(train_nnet.params[:-2], test_nnet.params[:-2]):
    assert p_train == p_test
    assert numpy.allclose(p_train.get_value(), p_test.get_value())

  return train_nnet, test_nnet

def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape, dtype=theano.config.floatX)
        return x * mask
    return x