import sys
import numpy
import os
from theano import tensor as T, function, printing
import nn_layers
import sgd_trainer
from tqdm import tqdm

def main():
    print "Hello"

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if not mode in ['TRAIN', 'TRAIN-ALL']:
            print "ERROR! The two possible training settings are: ['TRAIN', 'TRAIN-ALL']"
            sys.exit(1)

    print "Running training in the {} setting".format(mode)

    data_dir = mode

    if mode in ['TRAIN-ALL']:
        q_train = numpy.load(os.path.join(data_dir, 'train-all.questions.npy'))
        a_train = numpy.load(os.path.join(data_dir, 'train-all.answers.npy'))
        q_overlap_train = numpy.load(os.path.join(data_dir, 'train-all.q_overlap_indices.npy'))
        a_overlap_train = numpy.load(os.path.join(data_dir, 'train-all.a_overlap_indices.npy'))
        y_train = numpy.load(os.path.join(data_dir, 'train-all.labels.npy'))
    else:
        q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
        a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
        q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
        a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
        y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

    q_dev = numpy.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = numpy.load(os.path.join(data_dir, 'dev.answers.npy'))
    q_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
    a_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
    y_dev = numpy.load(os.path.join(data_dir, 'dev.labels.npy'))
    qids_dev = numpy.load(os.path.join(data_dir, 'dev.qids.npy'))

    q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
    q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
    a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
    y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))

    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]

    print 'q_train', q_train.shape
    print 'q_dev', q_dev.shape
    print 'q_test', q_test.shape

    print 'q_overlap_train', q_overlap_train.shape


    ndim = 5
    print "Generating random vocabulary for word overlap indicator features with dim:", ndim
    dummy_word_id = numpy.max(a_overlap_train)
    # vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
    print "Gaussian"
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id + 1, ndim) * 0.25
    # vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.05
    # vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
    vocab_emb_overlap[-1] = 0

    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_glove.twitter.27B.50d.txt.npy')

    print "Loading word embeddings from", fname
    vocab_emb = numpy.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(a_train)
    print "Word embedding matrix size:", vocab_emb.shape

    x = T.dmatrix('x')
    x_q = T.lmatrix('q')
    x_q_overlap = T.lmatrix('q_overlap')
    x_a = T.lmatrix('a')
    x_a_overlap = T.lmatrix('a_overlap')
    y = T.ivector('y')

    #######
    n_outs = 2

    n_epochs = 25
    batch_size = 50
    learning_rate = 0.1
    max_norm = 0

    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm

    ## 1st conv layer.
    ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]

    ### Nonlinearity type
    activation = T.tanh

    dropout_rate = 0.5
    nkernels = 100
    q_k_max = 1
    a_k_max = 1

    q_filter_widths = [5]

    ###### QUESTION ######
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths) - 1)
    lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths) - 1)

    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

    num_input_channels = 1
    input_shape = (batch_size, num_input_channels, q_max_sent_size + 2 * (max(q_filter_widths) - 1), ndim)
    q_logistic_n_in = nkernels * len(q_filter_widths) * q_k_max
    conv_layers = []
    for filter_width in q_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=q_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)
        hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=q_logistic_n_in, n_out=q_logistic_n_in, activation=activation)


    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_q = nn_layers.FeedForwardNet(layers=[
        lookup_table,
        join_layer,
        flatten_layer,
        hidden_layer,
    ])
    nnet_q.set_input((x_q, x_q_overlap))
    print nnet_q
    ######
    batch_x_q = T.lmatrix('batch_x_q')
    batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')

    inputs_pred = [batch_x_q,batch_x_q_overlap]
    givens_pred = {x_q: batch_x_q,x_q_overlap: batch_x_q_overlap}
    output = nnet_q.layers[-1].result

    output_fn = function(inputs=inputs_pred, outputs=output,givens=givens_pred)

    def output_batch(batch_iterator):
        preds = numpy.hstack([output_fn(batch_x_q, batch_x_q_overlap) for
                              batch_x_q, batch_x_q_overlap, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]



    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, q_overlap_train, y_train],batch_size=batch_size, randomize=True)

    o = output_batch(train_set_iterator)

    print(o.shape)


if __name__ == '__main__':
    main()
