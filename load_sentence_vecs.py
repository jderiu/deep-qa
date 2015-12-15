import cPickle
import nn_layers
import numpy
from theano import tensor as T
from theano import function,printing,sandbox
import os
from tqdm import tqdm
import sgd_trainer


CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"
def main():
    data_dir = HOME_DIR
    ##########
    # LAYERS #
    #########

    numpy_rng = numpy.random.RandomState(123)
    parameterMap = cPickle.load(open('parameters.p', 'rb'))
    filter_shape = parameterMap['filterShape']
    input_shape = parameterMap['inputShape']
    filter_width = parameterMap['filterWidth']
    activation = parameterMap['activation']
    q_logistic_n_in = parameterMap['qLogisticIn']
    k_max = parameterMap['kmax']

    tweets = T.imatrix('tweets_train')
    batch_tweets= T.imatrix('batch_x_q')

    print type(parameterMap['LookupTableFastStaticW'].get_value()[0][0])

    lookup_table_words = nn_layers.LookupTableFastStatic(
        W=parameterMap['LookupTableFastStaticW'].get_value(),
        pad=filter_width-1
    )

    conv = nn_layers.Conv2dLayer(
        W=parameterMap['Conv2dLayerW'],
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    non_linearity = nn_layers.NonLinearityLayer(
        b=parameterMap['NonLinearityLayerB'],
        b_size=filter_shape[0],
        activation=activation
    )

    pooling = nn_layers.KMaxPoolLayer(k_max=k_max)

    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[
        conv,
        non_linearity,
        pooling
    ])

    flatten_layer = nn_layers.FlattenLayer()

    hidden_layer = nn_layers.LinearLayer(
        W=parameterMap['LinearLayerW'],
        b=parameterMap['LinearLayerB'],
        rng=numpy_rng,
        n_in=q_logistic_n_in,
        n_out=q_logistic_n_in,
        activation=activation
    )

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table_words,
        conv2dNonLinearMaxPool,
        flatten_layer,
        hidden_layer
    ])

    nnet_tweets.set_input(tweets)
    print nnet_tweets

    #######################
    # Get Sentence Vectors#
    ######################

    batch_size = 50
    test_tweets = numpy.load(os.path.join(data_dir, 'all-merged.tweets.npy'))
    qids_test = numpy.load(os.path.join(data_dir, 'all-merged.tids.npy'))

    inputs_senvec = [batch_tweets]
    givents_senvec = {tweets:batch_tweets}

    output = nnet_tweets.layers[-1].output

    output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_tweets],
        batch_size=batch_size,
        randomize=False
    )

    counter = 0
    fname = open(os.path.join(data_dir,'twitter_sentence_vecs_loaded.txt'), 'w+')
    for i, tweet in enumerate(tqdm(test_set_iterator), 1):
        o = output_fn(tweet[0])
        for vec in o:
            fname.write(qids_test[counter])
            for el in numpy.nditer(vec):
                fname.write(" %f" % el)
            fname.write("\n")
            counter+=1
            if counter == test_set_iterator.n_samples:
                break


if __name__ == '__main__':
    main()
