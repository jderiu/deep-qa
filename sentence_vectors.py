import cPickle
import numpy
import os
import theano
from theano import tensor as T
from theano import function,printing,sandbox
import nn_layers
import sgd_trainer
from tqdm import tqdm
import time
from sklearn import metrics
from collections import Counter
import theano.sandbox.cuda.basic_ops
import h5py

CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"

def main():
    data_dir = HOME_DIR

    #smiley_set_tweets = numpy.load(os.path.join(data_dir, 'smiley_twets.tweets.npy'))
    #smiley_set_seniments = numpy.load(os.path.join(data_dir, 'smiley_twets.sentiments.npy'))
    hf = h5py.File(os.path.join(data_dir,'tweets.h5'),'r')
    smiley_set_tweets_data = hf.get('smiley_twets.tweets')
    smiley_set_seniments_data = hf.get('smiley_twets.sentiments')
    smiley_set_tweets=numpy.array(smiley_set_tweets_data)
    smiley_set_seniments=numpy.array(smiley_set_seniments_data)

    smiley_set_seniments=smiley_set_seniments.astype(int)
    smiley_set_seniments=numpy.asarray([0 if x==-1 else 1 for x in smiley_set_seniments])
    print smiley_set_seniments    

    smiley_set = zip(smiley_set_tweets,smiley_set_seniments)
    numpy.random.shuffle(smiley_set)
    smiley_set_tweets[:],smiley_set_seniments[:] = zip(*smiley_set)

    print type(smiley_set_tweets[0][0])
    print Counter(smiley_set_seniments)

    train_set = smiley_set_tweets[0 : int(len(smiley_set_tweets) * 0.95)]
    dev_set = smiley_set_tweets[int(len(smiley_set_tweets) * 0.95):int(len(smiley_set_tweets) * 1)]
    y_train_set = smiley_set_seniments[0 : int(len(smiley_set_seniments) * 0.95)]
    y_dev_set = smiley_set_seniments[int(len(smiley_set_seniments) * 0.95):int(len(smiley_set_seniments) * 1)]
    
    print "Length trains_set:", len(train_set)
    print "Length dev_set:", len(dev_set)
    print "Length y_trains_set:", len(y_train_set)
    print "Length y_dev_set:", len(y_dev_set)

    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = smiley_set_tweets.shape[1]

    # Load word2vec embeddings
    fname_wordembeddings = os.path.join(data_dir, 'emb_glove.twitter.27B.50d.txt.npy')

    print "Loading word embeddings from", fname_wordembeddings
    vocab_emb = numpy.load(fname_wordembeddings)
    print type(vocab_emb[0][0])
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(smiley_set_tweets)
    print "Word embedding matrix size:", vocab_emb.shape

    tweets = T.imatrix('tweets_train')
    y = T.lvector('y_train')

    #######
    n_outs = 2
    n_epochs = 5
    batch_size = 50
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
    k_max = 1
    num_input_channels = 1
    filter_width = 5
    q_logistic_n_in = nkernels * k_max

    input_shape = (
        batch_size,
        num_input_channels,
        q_max_sent_size + 2 * (filter_width - 1),
        ndim
    )

    filter_shape = (
        nkernels,
        num_input_channels,
        filter_width,
        ndim
    )

    ##########
    # LAYERS #
    #########

    parameterMap = {}
    parameterMap['filterShape'] = filter_shape
    parameterMap['inputShape'] = input_shape
    parameterMap['filterWidth'] = filter_width
    parameterMap['activation'] = activation
    parameterMap['qLogisticIn'] = q_logistic_n_in
    parameterMap['kmax'] = k_max


    lookup_table_words = nn_layers.LookupTableFastStatic(
        W=vocab_emb,
        pad=filter_width-1
    )

    parameterMap['LookupTableFastStaticW'] = lookup_table_words.W

    conv = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    parameterMap['Conv2dLayerW'] = conv.W

    non_linearity = nn_layers.NonLinearityLayer(
        b_size=filter_shape[0],
        activation=activation
    )

    parameterMap['NonLinearityLayerB'] = non_linearity.b

    pooling = nn_layers.KMaxPoolLayerNative(input_shape[2] - filter_width + 1)
    pooling = nn_layers.MaxPoolLayer1()

    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[
        conv,
        non_linearity,
        pooling
    ])

    flatten_layer = nn_layers.FlattenLayer()

    hidden_layer = nn_layers.LinearLayer(
        numpy_rng,
        n_in=q_logistic_n_in,
        n_out=q_logistic_n_in,
        activation=activation
    )

    parameterMap['LinearLayerW'] = hidden_layer.W
    parameterMap['LinearLayerB'] = hidden_layer.b

    classifier = nn_layers.LogisticRegression(n_in=q_logistic_n_in, n_out=n_outs)

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table_words,
        conv2dNonLinearMaxPool,
        flatten_layer,
        hidden_layer,
        classifier
    ])

    nnet_tweets.set_input(tweets)
    print nnet_tweets
    ######




    ################
    # TRAIN  MODEL #
    ###############
    ZEROUT_DUMMY_WORD = True

    batch_tweets= T.imatrix('batch_x_q')
    batch_y = T.lvector('batch_y')

    params = nnet_tweets.params
    cost = nnet_tweets.layers[-1].training_cost(y)
    predictions = nnet_tweets.layers[-1].y_pred
    predictions_prob = nnet_tweets.layers[-1].p_y_given_x[:, -1]

    inputs_train = [batch_tweets, batch_y]
    givens_train = {tweets: batch_tweets,
                    y: batch_y}

    inputs_pred = [batch_tweets]
    givens_pred = {tweets:batch_tweets}

    updates = sgd_trainer.get_adadelta_updates(
        cost,
        params,
        rho=0.95,
        eps=1e-6,
        max_norm=max_norm,
        word_vec_name='W_emb'
    )

    train_fn = theano.function(
        inputs=inputs_train,
        outputs=cost,
        updates=updates,
        givens=givens_train
    )

    pred_prob_fn = theano.function(
        inputs=inputs_pred,
        outputs=predictions_prob,
        givens=givens_pred
    )

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [train_set, y_train_set],
        batch_size=batch_size,
        randomize=True
    )

    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [dev_set],
        batch_size=batch_size,
        randomize=True)

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]


    if ZEROUT_DUMMY_WORD:
        W_emb_list = [w for w in params if w.name == 'W_emb']
        zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    best_dev_acc = -numpy.inf
    epoch = 0
    timer_train = time.time()
    no_best_dev_update = 0
    num_train_batches = len(train_set_iterator)
    while epoch < n_epochs:
        timer = time.time()
        for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator), 1):
            train_fn(tweet, y_label)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()

            if i % 2000 == 0 or i == num_train_batches:

                y_pred_dev = predict_prob_batch(dev_set_iterator)
                dev_acc = metrics.roc_auc_score(y_dev_set, y_pred_dev) * 100
                if dev_acc > best_dev_acc:
                    print('epoch: {} batch: {} dev auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc,best_dev_acc))
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                    no_best_dev_update = 0

        if no_best_dev_update >= 3:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)

    cPickle.dump(parameterMap, open('parameters.p', 'wb'))

    #######################
    # Get Sentence Vectors#
    ######################
    #test_tweets = numpy.load(os.path.join(data_dir, 'all-merged.tweets.npy'))
    #qids_test = numpy.load(os.path.join(data_dir, 'all-merged.tids.npy'))

    test_tweets_data = hf.get('all-merged.tweets')
    qids_test_data = hf.get('all-merged.tids')
    test_tweets = numpy.array(test_tweets_data)
    qids_test = numpy.array(qids_test_data)

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
    fname = open(os.path.join(data_dir,'twitter_sentence_vecs.txt'), 'w+')
    for i, (tweet) in enumerate(tqdm(test_set_iterator), 1):
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
