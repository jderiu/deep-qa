import cPickle
import numpy
import os
import theano
from theano import tensor as T
from theano import function
import nn_layers
import sgd_trainer
from tqdm import tqdm
import time
from sklearn import metrics
import getopt
import theano.sandbox.cuda.basic_ops
import sys
import math

CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"

def load_smiley_tweets(fname_ps,max_it=numpy.inf):
    tweet_set = numpy.load(fname_ps)
    it = 0
    while it <= max_it:
        try:
            batch = numpy.load(fname_ps)
        except:
            break
        tweet_set = numpy.concatenate((tweet_set,batch),axis=0)
        it += 1
    return tweet_set


def semeval_f1(y_truth,y_pred):
    neg_prec_up = 0
    neg_prec_down= 0
    neg_recall_up = 0
    neg_recall_down = 0

    pos_prec_up = 0
    pos_prec_down = 0
    pos_recall_up = 0
    pos_recall_down = 0

    for (target,prediction) in zip(y_truth,y_pred):
        if target == 0 and prediction == 0:
            neg_prec_up += 1
            neg_recall_up += 1
        if prediction == 0:
            neg_prec_down += 1
        if target == 0:
            neg_recall_down += 1

        if prediction == 2 and target == 2:
            pos_prec_up += 1
            pos_recall_up += 1
        if prediction == 2:
            pos_prec_down += 1
        if target == 2:
            pos_recall_down += 1

    if neg_prec_down == 0:
        neg_precision = 1.0
    else:
        neg_precision = 1.0*neg_prec_up/neg_prec_down

    if pos_prec_down == 0:
        pos_precision = 1.0
    else:
        pos_precision = 1.0*pos_prec_up/pos_prec_down

    if neg_recall_down == 0:
        neg_recall = 1.0
    else:
        neg_recall = 1.0*neg_recall_up/neg_recall_down

    if pos_recall_down == 0:
        pos_recall = 1.0
    else:
        pos_recall = 1.0*pos_recall_up/pos_recall_down

    if (neg_recall + neg_precision) == 0:
        neg_F1 = 0.0
    else:
        neg_F1 = 2*(neg_precision*neg_recall)/(neg_precision + neg_recall)

    if (pos_recall + pos_precision) == 0:
        pos_F1 = 0.0
    else:
        pos_F1 = 2*(pos_precision*pos_recall)/(pos_precision + pos_recall)

    f1 = (neg_F1 + pos_F1)/2
    return f1


def training(nnet,train_set_iterator,train_fn):
    params = nnet.params

    W_emb_list = [w for w in params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    num_train_batches = len(train_set_iterator)
    for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
        train_fn(tweet, y_label)

        # Make sure the null word in the word embeddings always remains zero
        zerout_dummy_word()


def get_next_chunk_pos_neg(pos_file, neg_file, n_chunks=1):
    tweet_set = None
    sentiment_set = None
    it = 0
    while True:
        try:
            batch_pos = numpy.load(pos_file)
            batch_neg = numpy.load(neg_file)
            batch = numpy.concatenate((batch_pos,batch_neg),axis=0)

            pos_len = batch_pos.shape[0]
            neg_len = batch_neg.shape[0]
            sentiment = numpy.concatenate((numpy.ones(pos_len),numpy.zeros(neg_len)),axis=0)
            if tweet_set == None:
                tweet_set = batch
                sentiment_set = sentiment
            else:
                tweet_set = numpy.concatenate((tweet_set,batch),axis=0)
                sentiment_set = numpy.concatenate((sentiment_set,sentiment),axis=0)
        except:
            break
        it += 1
        if not (it < n_chunks):
            break

    return tweet_set,sentiment_set,it


def get_next_chunk(fname_tweet,fname_sentiment,n_chunks=1):
    tweet_set = None
    sentiment_set = None
    it = 0
    while True:
        try:
            batch_tweet = numpy.load(fname_tweet)
            batch_sentiment = numpy.load(fname_sentiment)
            if tweet_set == None:
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


def usage():
    print 'python parse_tweets.py -i <small,30M> -e <embedding:glove or custom>'


def main():
    timestamp = str(long(time.time()*1000))
    input_fname = 'small'
    embedding = 'glove'
    etype = 'small'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:t:", ["help", "input=","embedding=","type="])
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
        elif o in ("-t", "--type"):
            etype = a
        else:
            assert False, "unhandled option"

    data_dir = HOME_DIR + '_' + input_fname

    numpy_rng = numpy.random.RandomState(123)

    q_max_sent_size = 140

    # Load word2vec embeddings
    if embedding == 'glove':
        embedding_fname = 'emb_glove.twitter.27B.50d.txt.npy'
    else:
        embedding_fname = 'emb_smiley_tweets_embedding_{}.npy'.format(etype)

    fname_wordembeddings = os.path.join(data_dir, embedding_fname)

    #parameter_map = cPickle.load(open(data_dir+'/parameters_distant_f166.p', 'rb'))


    print "Loading word embeddings from", fname_wordembeddings
    vocab_emb = numpy.load(fname_wordembeddings)
    #vocab_emb = parameter_map['LookupTableFastStaticW'].get_value()
    print type(vocab_emb[0][0])
    print "Word embedding matrix size:", vocab_emb.shape

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

    dropout_rate = 0.5
    nkernels1 = 200
    nkernels2 = 200
    k_max = 1
    shape1 = 4
    st = None
    num_input_channels = 1
    filter_width1 = 3
    filter_width2 = 3
    q_logistic_n_in = nkernels1 * k_max
    sent_size = q_max_sent_size + 2*(filter_width1 - 1)
    layer1_size = (sent_size - filter_width1 + 1 - shape1)//st[0] + 1
    print layer1_size
    #n_in = nkernels2*math.ceil((layer1_size - filter_width + 1)/shape2)
    #n_in = nkernels2*k_max

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

    lookup_table_words = nn_layers.LookupTableFast(
        W=vocab_emb,
        pad=filter_width1-1
    )

    parameter_map['LookupTableFastStaticW'] = lookup_table_words.W

    conv_layers = []
    filter_shape = (
        nkernels1,
        num_input_channels,
        filter_width1,
        ndim
    )

    parameter_map['FilterShape' + str(filter_width1)] = filter_shape

    conv = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    parameter_map['Conv2dLayerW' + str(filter_width1)] = conv.W

    non_linearity = nn_layers.NonLinearityLayer(
        b_size=filter_shape[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB' + str(filter_width1)] = non_linearity.b

    pooling = nn_layers.KMaxPoolLayerNative(shape=shape1,ignore_border=True,st=st)

    parameter_map['PoolingShape1'] = shape1
    parameter_map['PoolingSt1'] = st

    input_shape2 = (
        batch_size,
        nkernels1,
        (input_shape[2] - filter_width1 + 1 - shape1)//st[0] + 1,
        1
    )

    parameter_map['input_shape2'+ str(filter_width1)] = input_shape2

    filter_shape2 = (
        nkernels2,
        nkernels1,
        filter_width2,
        1
    )

    parameter_map['FilterShape2' + str(filter_width1)] = filter_shape2

    con2 = nn_layers.Conv2dLayer(
        rng=numpy_rng,
        input_shape=input_shape2,
        filter_shape=filter_shape2
    )

    parameter_map['Conv2dLayerW2' + str(filter_width1)] = con2.W

    non_linearity2 = nn_layers.NonLinearityLayer(
        b_size=filter_shape2[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB2' + str(filter_width1)] = non_linearity2.b

    shape2 = input_shape2[2] - filter_width2 + 1
    pooling2 = nn_layers.KMaxPoolLayerNative(shape=shape2,ignore_border=True)
    n_in = nkernels2*(layer1_size - filter_width2 + 1)//shape2
    parameter_map['n_in'] = n_in
    parameter_map['PoolingShape2'] = shape2

    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[
        conv,
        non_linearity,
        pooling,
        con2,
        non_linearity2,
        pooling2
    ])
    conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
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
        join_layer,
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
    n_epochs = 2
    early_stop = 3
    best_dev_acc = -numpy.inf
    no_best_dev_update = 0
    timer_train = time.time()
    done = False
    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
    while epoch < n_epochs and not done:
        max_chunks = numpy.inf
        curr_chunks = 0
        timer = time.time()
        #fname_ps = open(os.path.join(data_dir, 'smiley_tweets_pos_{}.tweets.npy'.format(embedding)),'rb')
        #fname_neg = open(os.path.join(data_dir, 'smiley_tweets_neg_{}.tweets.npy'.format(embedding)),'rb')
        fname_tweet = open(os.path.join(data_dir, 'smiley_tweets_{}.tweets.npy'.format(embedding)),'rb')
        fname_sentiments = open(os.path.join(data_dir, 'smiley_tweets_{}.sentiments.npy'.format(embedding)),'rb')
        while curr_chunks < max_chunks:
            smiley_set_tweets,smiley_set_sentiments,chunks = get_next_chunk(fname_tweet, fname_sentiments, n_chunks=4)
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

            num_train_batches = len(train_set_iterator)
            for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
                train_fn(tweet, y_label)

            # Make sure the null word in the word embeddings always remains zero
            zerout_dummy_word()

            y_pred_dev = predict_batch(dev_set_iterator)
            dev_acc = metrics.accuracy_score(y_dev_set, y_pred_dev) * 100

            if dev_acc > best_dev_acc:
                    print('epoch: {} chunk: {} best_chunk_auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, curr_chunks, dev_acc,best_dev_acc))
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                    no_best_dev_update = 0
            else:
                print('epoch: {} chunk: {} best_chunk_auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, curr_chunks, dev_acc,best_dev_acc))
            cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format('distant'), 'wb'))

        cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format('distant'), 'wb'))
        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))

        if no_best_dev_update >= early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break
        no_best_dev_update += 1
        epoch += 1
        fname_tweet.close()
        fname_sentiments.close()

    cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format('distant'), 'wb'))
    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        #params[i].set_value(param, borrow=True)
        pass

    only_distant = True

    if only_distant:
        return
    #######################
    # Supervised Learining#
    ######################

    training_tweets = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.tweets.npy'.format(embedding)))
    training_sentiments = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.sentiments.npy'.format(embedding)))

    dev_tweets = numpy.load(os.path.join(data_dir, 'twitter-test-gold-B.downloaded_{}.tweets.npy'.format(embedding)))
    dev_sentiments = numpy.load(os.path.join(data_dir, 'twitter-test-gold-B.downloaded_{}.sentiments.npy'.format(embedding)))

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
    params = nnet_tweets.params
    print nnet_tweets
    cost = nnet_tweets.layers[-1].training_cost(y)

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

    epoch = 0
    n_epochs = 100
    early_stop = 10
    check_freq = train_set_iterator.n_batches/10
    timer_train = time.time()
    no_best_dev_update = 0
    best_dev_acc = -numpy.inf
    num_train_batches = len(train_set_iterator)
    while epoch < n_epochs:
        timer = time.time()
        for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
                train_fn(tweet, y_label)

                if i % check_freq == 0 or i == num_train_batches:
                    y_pred_dev = predict_batch(dev_set_iterator)
                    dev_acc = semeval_f1(dev_sentiments,y_pred_dev)*100
                    if dev_acc > best_dev_acc:
                        print('epoch: {} chunk: {} best_chunk_f1: {:.4f}; best_dev_f1: {:.4f}'.format(epoch, i, dev_acc,best_dev_acc))
                        best_dev_acc = dev_acc
                        best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                        no_best_dev_update = 0
                        cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format('supervised'), 'wb'))


        zerout_dummy_word()
        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1
        if no_best_dev_update >= early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)


    #######################
    # Get Sentence Vectors#
    ######################
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
    fname = open(os.path.join(data_dir,'twitter_sentence_vecs_{}.txt'.format(timestamp)), 'w+')
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
