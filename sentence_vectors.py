import sys
import numpy
import os
from theano import tensor as T, function, printing
import nn_layers
import sgd_trainer
from tqdm import tqdm

def main():
    data_dir = "semeval"
    q_train = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev.tweets.npy'))
    qids_test = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev.tids.npy'))



    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]


    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_glove.twitter.27B.50d.txt.npy')

    print "Loading word embeddings from", fname
    vocab_emb = numpy.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(q_train)
    print "Word embedding matrix size:", vocab_emb.shape

    x_q = T.lmatrix('q')

    #######
    n_outs = 2

    n_epochs = 25
    batch_size = q_train.shape[0]
    learning_rate = 0.1
    max_norm = 0

    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm

    ## 1st conv layer.
    ndim = vocab_emb.shape[1]

    ### Nonlinearity type
    activation = T.tanh

    dropout_rate = 0.5
    nkernels = 100
    q_k_max = 1
    a_k_max = 1

    q_filter_widths = [5]

    ###### QUESTION ######
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths) - 1)


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
        lookup_table_words,
        join_layer,
        flatten_layer,
        hidden_layer,
    ])
    nnet_q.set_input(x_q)
    print nnet_q
    ######
    batch_x_q = T.lmatrix('batch_x_q')

    inputs_pred = [batch_x_q]
    givens_pred = {x_q: batch_x_q}
    output = nnet_q.layers[-1].result

    output_fn = function(inputs=inputs_pred, outputs=output,givens=givens_pred)

    def output_batch(batch_iterator):
        preds = numpy.hstack([output_fn(batch_x_q) for
                              batch_x_q, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]



    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train],batch_size=batch_size, randomize=True)

    for tweet in q_train:
        print output_fn(tweet)


    print o.shape



if __name__ == '__main__':
    main()
