import cPickle
import nn_layers
import numpy
from theano import tensor as T
from theano import function
import theano
import os
from tqdm import tqdm
import sgd_trainer
import sys
import sentence_vectors as sv
import time
import getopt


def usage():
    print 'python load_sentence_vecs.py -i <small,30M> -e <embedding:glove or custom>'


def main():
    ##########
    # LAYERS #
    #########

    HOME_DIR = "semeval_parsed"
    timestamp = str(long(time.time()*1000))
    input_fname = 'small'
    embedding = 'glove'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:", ["help", "input=","embedding="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-e","--embedding"):
            if a in ('glove','custom'):
                embedding = a
            else:
                usage()
                sys.exit()
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            input_fname = a
        else:
            assert False, "unhandled option"

    data_dir = HOME_DIR + '_' + input_fname
    numpy_rng = numpy.random.RandomState(123)
    parameterMap = cPickle.load(open(data_dir+'/parameters.p', 'rb'))
    input_shape = parameterMap['inputShape']
    filter_widths = parameterMap['filterWidths']
    activation = parameterMap['activation']
    q_logistic_n_in = parameterMap['qLogisticIn']
    k_max = parameterMap['kmax']

    tweets = T.imatrix('tweets_train')
    y = T.lvector('y')
    batch_tweets= T.imatrix('batch_x_q')
    batch_y = T.lvector('batch_y')

    print type(parameterMap['LookupTableFastStaticW'].get_value()[0][0])

    lookup_table_words = nn_layers.LookupTableFastStatic(
        W=parameterMap['LookupTableFastStaticW'].get_value(),
        pad=max(filter_widths)-1
    )

    conv_layers = []
    for filter_width in filter_widths:

        conv = nn_layers.Conv2dLayer(
            W=parameterMap['Conv2dLayerW' + str(filter_width)],
            rng=numpy_rng,
            filter_shape=parameterMap['FilterShape' + str(filter_width)],
            input_shape=input_shape
        )

        non_linearity = nn_layers.NonLinearityLayer(
            b=parameterMap['NonLinearityLayerB' + str(filter_width)],
            b_size=parameterMap['FilterShape' + str(filter_width)][0],
            activation=activation
        )

        pooling = nn_layers.KMaxPoolLayer(k_max=k_max)

        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[
            conv,
            non_linearity,
            pooling
        ])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
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
        join_layer,
        flatten_layer,
        hidden_layer
    ])

    nnet_tweets.set_input(tweets)
    print nnet_tweets


     #######################
    # Supervised Learining#
    ######################
    batch_size = 500

    training_tweets = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.tweets.npy'.format(embedding)))
    training_sentiments = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.sentiments.npy'.format(embedding)))

    dev_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter_{}.tweets.npy'.format(embedding)))
    dev_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter_{}.sentiments.npy'.format(embedding)))

    inputs_train = [batch_tweets, batch_y]
    givens_train = {tweets: batch_tweets,
                    y: batch_y}

    inputs_pred = [batch_tweets]
    givens_pred = {tweets:batch_tweets}

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [training_tweets, training_sentiments],
        batch_size=batch_size,
        randomize=True
    )

    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [dev_tweets],
        batch_size=batch_size,
        randomize=False
    )

    n_outs = 3
    classifier = nn_layers.LogisticRegression(n_in=q_logistic_n_in, n_out=n_outs)

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table_words,
        join_layer,
        flatten_layer,
        hidden_layer,
        classifier
    ])


    nnet_tweets.set_input(tweets)
    print nnet_tweets

    params = nnet_tweets.params
    cost = nnet_tweets.layers[-1].training_cost(y)
    predictions = nnet_tweets.layers[-1].y_pred

    updates = sgd_trainer.get_adadelta_updates(
        cost,
        params,
        rho=0.95,
        eps=1e-6,
        max_norm=0,
        word_vec_name='None'
    )

    train_fn = theano.function(
        inputs=inputs_train,
        outputs=cost,
        updates=updates,
        givens=givens_train
    )

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]

    n_epochs = 100
    check_freq = train_set_iterator.n_batches/10
    sv.training(nnet_tweets,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_batch,dev_sentiments,data_dir=data_dir,parameter_map=parameterMap,n_outs=3,early_stop=10,check_freq=check_freq)


    #######################
    # Get Sentence Vectors#
    ######################

    batch_size = input_shape[0]
    test_tweets = numpy.load(os.path.join(data_dir, 'all_merged_{}.tweets.npy'.format(embedding)))
    qids_test = numpy.load(os.path.join(data_dir, 'all_merged_{}.tids.npy'.format(embedding)))

    inputs_senvec = [batch_tweets]
    givents_senvec = {tweets:batch_tweets}


    output = nnet_tweets.layers[-2].output

    output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_tweets],
        batch_size=batch_size,
        randomize=False
    )

    counter = 0
    fname = open(os.path.join(data_dir,'twitter_sentence_vecs_loaded_{}.txt'.format(timestamp)), 'w+')
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
