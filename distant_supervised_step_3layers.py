import cPickle
import numpy
import os
import theano
from theano import tensor as T
import nn_layers
import sgd_trainer
from tqdm import tqdm
import time
from sklearn import metrics
from scipy.stats import truncnorm
import theano.sandbox.cuda.basic_ops
import sys
import getopt


def get_next_chunk(fname_tweet,fname_sentiment,n_chunks=1):
    tweet_set = None
    sentiment_set = None
    it = 0
    while True:
        try:
            batch_tweet = numpy.load(fname_tweet)
            batch_sentiment = numpy.load(fname_sentiment)
            if tweet_set is None:
                tweet_set = batch_tweet
                sentiment_set = batch_sentiment
            else:
                tweet_set = numpy.concatenate((tweet_set,batch_tweet),axis=0)
                sentiment_set = numpy.concatenate((sentiment_set,batch_sentiment),axis=0)
        except:
            break
        it += 1
        if not (it < n_chunks):
            break

    return tweet_set,sentiment_set,it


def main(argv):
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    test_type = ''
    n_chunks = numpy.inf
    ndim = 52
    randtype = 'uniform'
    update_wemb = True

    argv = map(lambda x: x.replace('\r',''),argv)
    try:
      opts, args = getopt.getopt(argv,"ut:r:d:c:",["testtype=","randtype=","ndim=","n_chunks="])
    except getopt.GetoptError as e:
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-r", "--randtype"):
            randtype = arg
        elif opt in ("-d", "--ndim"):
            ndim = int(arg)
        elif opt in ("-t", "--testtype"):
            test_type = arg
        elif opt in ("-c", "--n_chunks"):
            n_chunks = int(arg)
        elif opt == "-u":
            update_wemb = True

    print test_type
    print n_chunks
    print ndim
    print randtype
    print update_wemb

    data_dir = HOME_DIR + '_' + input_fname
    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = 140

    # Load word2vec embeddings
    embedding_fname = 'emb_smiley_tweets_embedding_final.npy'
    fname_wordembeddings = os.path.join(data_dir, embedding_fname)
    print "Loading word embeddings from", fname_wordembeddings
    vocab_emb = numpy.load(fname_wordembeddings)
    print type(vocab_emb[0][0])
    print "Word embedding matrix size:", vocab_emb.shape

    if randtype == 'uniform':
        dim1 = vocab_emb.shape[0]
        dim2 = ndim
        vocab_emb = (numpy_rng.randn(dim1, dim2) * 0.25).astype(numpy.float32)
    elif randtype == 'truncnorm':
        dim1 = vocab_emb.shape[0]
        dim2 = ndim
        vocab_emb = truncnorm.rvs(-1, 1,scale=0.8,size=(dim1,dim2)).astype(numpy.float32)

    tweets = T.imatrix('tweets_train')
    y = T.lvector('y_train')

    #######
    n_outs = 2
    batch_size = 1000
    max_norm = 0

    print 'batch_size', batch_size
    print 'max_norm', max_norm

    ## 1st conv layer.
    ndim = vocab_emb.shape[1]

    ### Nonlinearity type
    def relu(x):
        return x * (x > 0)

    activation = relu
    nkernels1 = 200
    nkernels2 = 200
    nkernels3 = 200
    k_max = 1
    shape1 = 6
    st = (2,1)
    shape2 = 4
    st2 = (1,1)
    num_input_channels = 1
    filter_width1 = 6
    filter_width2 = 4
    filter_width3 = 4
    q_logistic_n_in = nkernels1 * k_max
    sent_size = q_max_sent_size + 2*(filter_width1 - 1)
    layer1_size = (sent_size - filter_width1 + 1 - shape1)//st[0] + 1

    input_shape = (
        batch_size,
        num_input_channels,
        q_max_sent_size + 2 * (filter_width1 - 1),
        ndim
    )

    ##########
    # LAYERS #
    #########
    parameter_map = {}
    parameter_map['nKernels1'] = nkernels1
    parameter_map['nKernels2'] = nkernels2
    parameter_map['num_input_channels'] = num_input_channels
    parameter_map['ndim'] = ndim
    parameter_map['inputShape'] = input_shape
    parameter_map['activation'] = 'relu'
    parameter_map['qLogisticIn'] = q_logistic_n_in
    parameter_map['kmax'] = k_max
    parameter_map['st'] = st

    parameter_map['filterWidth'] = filter_width1

    if not update_wemb:
        lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb,pad=filter_width1-1)
    else:
        lookup_table_words = nn_layers.LookupTableFast(W=vocab_emb,pad=filter_width1-1)

    parameter_map['LookupTableFastStaticW'] = lookup_table_words.W

    #conv_layers = []
    filter_shape = (
        nkernels1,
        num_input_channels,
        filter_width1,
        ndim
    )

    parameter_map['FilterShape'] = filter_shape

    conv = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    parameter_map['Conv2dLayerW'] = conv.W

    non_linearity = nn_layers.NonLinearityLayer(
        b_size=filter_shape[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB'] = non_linearity.b

    pooling = nn_layers.KMaxPoolLayerNative(shape=shape1,ignore_border=True,st=st)

    parameter_map['PoolingShape1'] = shape1
    parameter_map['PoolingSt1'] = st

    input_shape2 = (
        batch_size,
        nkernels1,
        (input_shape[2] - filter_width1 + 1 - shape1)//st[0] + 1,
        1
    )

    parameter_map['input_shape2'] = input_shape2

    filter_shape2 = (
        nkernels2,
        nkernels1,
        filter_width2,
        1
    )

    parameter_map['FilterShape2'] = filter_shape2

    con2 = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        input_shape=input_shape2,
        filter_shape=filter_shape2
    )

    parameter_map['Conv2dLayerW2'] = con2.W

    non_linearity2 = nn_layers.NonLinearityLayer(
        b_size=filter_shape2[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB2'] = non_linearity2.b

    pooling2 = nn_layers.KMaxPoolLayerNative(shape=shape2,st=st2,ignore_border=True)
    parameter_map['PoolingShape2'] = shape2
    parameter_map['st2'] = st2

    #layer 3
    input_shape3 = (
        batch_size,
        nkernels2,
        (input_shape2[2] - filter_width2 + 1 - shape2)//st2[0] + 1,
        1
    )

    parameter_map['input_shape3'] = input_shape3

    filter_shape3 = (
        nkernels3,
        nkernels2,
        filter_width3,
        1
    )

    parameter_map['FilterShape3'] = filter_shape3

    con3 = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        input_shape=input_shape3,
        filter_shape=filter_shape3
    )

    parameter_map['Conv2dLayerW3'] = con3.W

    non_linearity3 = nn_layers.NonLinearityLayer(
        b_size=filter_shape3[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB3'] = non_linearity3.b

    shape3 = input_shape3[2] - filter_width3 + 1
    pooling3 = nn_layers.KMaxPoolLayerNative(shape=shape3,ignore_border=True)
    parameter_map['PoolingShape3'] = shape3

    n_in = nkernels3*(input_shape3[2] - filter_width3 + 1)//shape3
    parameter_map['n_in'] = n_in

    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[
        conv,
        non_linearity,
        pooling,
        con2,
        non_linearity2,
        pooling2,
        con3,
        non_linearity3,
        pooling3
    ])

    flatten_layer = nn_layers.FlattenLayer()

    hidden_layer = nn_layers.LinearLayer(
        numpy_rng,
        n_in=n_in,
        n_out=n_in,
        activation=activation
    )

    parameter_map['LinearLayerW'] = hidden_layer.W
    parameter_map['LinearLayerB'] = hidden_layer.b

    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table_words,
        conv2dNonLinearMaxPool,
        flatten_layer,
        hidden_layer,
        classifier
    ])

    nnet_tweets.set_input(tweets)
    print nnet_tweets

    ################
    # TRAIN  MODEL #
    ###############

    batch_tweets= T.imatrix('batch_x_q')
    batch_y = T.lvector('batch_y')

    params = nnet_tweets.params
    print params
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

    pred_prob_fn = theano.function(
        inputs=inputs_pred,
        outputs=predictions_prob,
        givens=givens_pred
    )

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]

    W_emb_list = [w for w in params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    epoch = 0
    n_epochs = 1
    early_stop = 3
    best_dev_acc = -numpy.inf
    no_best_dev_update = 0
    timer_train = time.time()
    done = False
    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
    while epoch < n_epochs and not done:
        max_chunks = n_chunks
        curr_chunks = 0
        timer = time.time()
        fname_tweet = open(os.path.join(data_dir, 'smiley_tweets.tweets.npy'),'rb')
        fname_sentiments = open(os.path.join(data_dir, 'smiley_tweets.sentiments.npy'),'rb')
        while curr_chunks < max_chunks:
            smiley_set_tweets,smiley_set_sentiments,chunks = get_next_chunk(fname_tweet, fname_sentiments, n_chunks=2)
            print smiley_set_sentiments
            curr_chunks += chunks
            if smiley_set_tweets == None:
                break

            print 'Chunk number:',curr_chunks
            smiley_set_sentiments = smiley_set_sentiments.astype(int)

            smiley_set = zip(smiley_set_tweets,smiley_set_sentiments)
            numpy_rng.shuffle(smiley_set)
            smiley_set_tweets[:],smiley_set_sentiments[:] = zip(*smiley_set)

            train_set = smiley_set_tweets[0 : int(len(smiley_set_tweets) * 0.98)]
            dev_set = smiley_set_tweets[int(len(smiley_set_tweets) * 0.98):int(len(smiley_set_tweets) * 1)]
            y_train_set = smiley_set_sentiments[0 : int(len(smiley_set_sentiments) * 0.98)]
            y_dev_set = smiley_set_sentiments[int(len(smiley_set_sentiments) * 0.98):int(len(smiley_set_sentiments) * 1)]

            print "Length trains_set:", len(train_set)
            print "Length dev_set:", len(dev_set)
            print "Length y_trains_set:", len(y_train_set)
            print "Length y_dev_set:", len(y_dev_set)

            train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng,[train_set, y_train_set],batch_size=batch_size,randomize=True)

            dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng,[dev_set],batch_size=batch_size,randomize=False)

            for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
                train_fn(tweet, y_label)

            # Make sure the null word in the word embeddings always remains zero
            zerout_dummy_word()

            y_pred_dev = predict_batch(dev_set_iterator)
            dev_acc = metrics.accuracy_score(y_dev_set, y_pred_dev) * 100

            if dev_acc > best_dev_acc:
                    print('epoch: {} chunk: {} best_chunk_auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, curr_chunks, dev_acc,best_dev_acc))
                    best_dev_acc = dev_acc
                    no_best_dev_update = 0
            else:
                print('epoch: {} chunk: {} best_chunk_auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, curr_chunks, dev_acc,best_dev_acc))
            cPickle.dump(parameter_map, open(data_dir+'/parameters_{}_{}.p'.format('distant',test_type), 'wb'))

        cPickle.dump(parameter_map, open(data_dir+'/parameters_{}_{}.p'.format('distant',test_type), 'wb'))
        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))

        if no_best_dev_update >= early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break
        no_best_dev_update += 1
        epoch += 1
        fname_tweet.close()
        fname_sentiments.close()

    cPickle.dump(parameter_map, open(data_dir+'/parameters_{}_{}.p'.format('distant',test_type), 'wb'))
    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))


if __name__ == '__main__':
    main(sys.argv[1:])
