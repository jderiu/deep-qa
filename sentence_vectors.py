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
    negPrecUp = 0
    negPrecDown = 0
    negRecallUp = 0
    negRecallDown = 0

    posPrecUp = 0
    posPrecDown = 0
    posRecallUp = 0
    posRecallDown = 0

    for (target,prediction) in zip(y_truth,y_pred):
        if target == 0 and prediction == 0:
            negPrecUp += 1
            negRecallUp += 1
        if prediction == 0:
            negPrecDown += 1
        if target == 0:
            negRecallDown += 1

        if prediction == 2 and target == 2:
            posPrecUp += 1
            posRecallUp += 1
        if prediction == 2:
            posPrecDown += 1
        if target == 2:
            posRecallDown += 1

    if negPrecDown == 0:
        negPrecision = 1.0
    else:
        negPrecision = 1.0*negPrecUp/negPrecDown

    if posPrecDown == 0:
        posPrecision = 1.0
    else:
        posPrecision = 1.0*posPrecUp/posPrecDown

    if negRecallDown == 0:
        negRecall = 1.0
    else:
        negRecall = 1.0*negRecallUp/negRecallDown

    if posRecallDown == 0:
        posRecall = 1.0
    else:
        posRecall = 1.0*posRecallUp/posRecallDown

    if (negRecall + negPrecision) == 0:
        negF1 = 0.0
    else:
        negF1 = 2*(negPrecision*negRecall)/(negPrecision + negRecall)

    if (posRecall + posPrecision) == 0:
        posF1 = 0.0
    else:
        posF1 = 2*(posPrecision*posRecall)/(posPrecision + posRecall)

    f1 = (negF1 + posF1)/2
    return f1

def training(nnet,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_prob_batch,y_dev_set,data_dir,parameter_map,n_outs=2,early_stop=5,check_freq=300,sup_type='distant'):
    params = nnet.params
    ZEROUT_DUMMY_WORD = True

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
        for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
            train_fn(tweet, y_label)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()

            if i % check_freq == 0 or i == num_train_batches:
                y_pred_dev = predict_prob_batch(dev_set_iterator)
                if n_outs == 2:
                    dev_acc = metrics.roc_auc_score(y_dev_set, y_pred_dev) * 100
                else:
                    dev_acc = semeval_f1(y_dev_set,y_pred_dev)*100
                    dev_acc_c = metrics.accuracy_score(y_dev_set,y_pred_dev)*100
                if dev_acc > best_dev_acc:
                    print('epoch: {} batch: {} dev auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc,best_dev_acc))
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                    no_best_dev_update = 0
                    cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format(sup_type), 'wb'))

        if no_best_dev_update >= early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)
    params


def get_next_chunck(pos_file,neg_file,n_chunchks=1):
    tweet_set = None
    sentiment_set = None
    it = 0
    while True:
        try:
            batch_pos = numpy.load(pos_file)
            batch_neg = numpy.load(neg_file)
            batch = numpy.concatenate((batch_pos,batch_neg),axis=0)
            print batch.shape

            pos_len = batch_pos.shape[0]
            neg_len = batch_neg.shape[0]
            sentiment = numpy.concatenate((numpy.ones(pos_len),numpy.zeros(neg_len)),axis=0)
            if tweet_set == None:
                tweet_set = batch
                sentiment_set = sentiment
                print tweet_set.shape
            else:
                print tweet_set.shape
                tweet_set = numpy.concatenate((tweet_set,batch),axis=0)
                sentiment_set = numpy.concatenate((sentiment_set,sentiment),axis=0)
        except:
            break
        it += 1
        print it
        if not (it < n_chunchks):
            break

    return tweet_set,sentiment_set


def usage():
    print 'python parse_tweets.py -i <small,30M> -e <embedding:glove or custom>'


def main():
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

    fname_ps = open(os.path.join(data_dir, 'smiley_tweets_pos_{}.tweets.npy'.format(embedding)),'rb')
    fname_neg = open(os.path.join(data_dir, 'smiley_tweets_neg_{}.tweets.npy'.format(embedding)),'rb')
    numpy_rng = numpy.random.RandomState(123)

    q_max_sent_size = 140

    # Load word2vec embeddings
    if embedding == 'glove':
        embedding_fname = 'emb_glove.twitter.27B.50d.txt.npy'
    else:
        embedding_fname = 'emb_smiley_tweets_embedding_{}.npy'.format(input_fname)

    fname_wordembeddings = os.path.join(data_dir, embedding_fname)

    print "Loading word embeddings from", fname_wordembeddings
    vocab_emb = numpy.load(fname_wordembeddings)
    print type(vocab_emb[0][0])
    print "Word embedding matrix size:", vocab_emb.shape

    tweets = T.imatrix('tweets_train')
    y = T.lvector('y_train')

    #######
    n_outs = 2
    n_epochs = 1
    batch_size = 500
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
    nkernels = 300
    k_max = 1
    num_input_channels = 1
    filter_widths = [5]
    q_logistic_n_in = nkernels * k_max * len(filter_widths)

    input_shape = (
        batch_size,
        num_input_channels,
        q_max_sent_size + 2 * (max(filter_widths) - 1),
        ndim
    )



    ##########
    # LAYERS #
    #########

    parameter_map = {}
    parameter_map['nKernels'] = nkernels
    parameter_map['num_input_channels'] = num_input_channels
    parameter_map['ndim'] = ndim
    parameter_map['inputShape'] = input_shape
    parameter_map['filterWidths'] = filter_widths
    parameter_map['activation'] = activation
    parameter_map['qLogisticIn'] = q_logistic_n_in
    parameter_map['kmax'] = k_max

    lookup_table_words = nn_layers.LookupTableFastStatic(
        W=vocab_emb,
        pad=max(filter_widths)-1
    )

    parameter_map['LookupTableFastStaticW'] = lookup_table_words.W

    conv_layers = []
    for filter_width in filter_widths:
        filter_shape = (
            nkernels,
            num_input_channels,
            filter_width,
            ndim
        )

        parameter_map['FilterShape' + str(filter_width)] = filter_shape

        conv = nn_layers.Conv2dLayer(
            rng=numpy_rng,
            filter_shape=filter_shape,
            input_shape=input_shape
        )

        parameter_map['Conv2dLayerW' + str(filter_width)] = conv.W

        non_linearity = nn_layers.NonLinearityLayer(
            b_size=filter_shape[0],
            activation=activation
        )

        parameter_map['NonLinearityLayerB' + str(filter_width)] = non_linearity.b

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
        numpy_rng,
        n_in=q_logistic_n_in,
        n_out=q_logistic_n_in,
        activation=activation
    )

    parameter_map['LinearLayerW'] = hidden_layer.W
    parameter_map['LinearLayerB'] = hidden_layer.b

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
    ######




    ################
    # TRAIN  MODEL #
    ###############
    ZEROUT_DUMMY_WORD = True

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

    batch_number = 1
    while True:
        smiley_set_tweets,smiley_set_seniments = get_next_chunck(fname_ps,fname_neg,n_chunchks=4)
        if smiley_set_tweets == None:
            break
        print 'Chunk number:',batch_number
        smiley_set_seniments=smiley_set_seniments.astype(int)

        smiley_set = zip(smiley_set_tweets,smiley_set_seniments)
        numpy_rng.shuffle(smiley_set)
        smiley_set_tweets[:],smiley_set_seniments[:] = zip(*smiley_set)

        train_set = smiley_set_tweets[0 : int(len(smiley_set_tweets) * 0.98)]
        dev_set = smiley_set_tweets[int(len(smiley_set_tweets) * 0.98):int(len(smiley_set_tweets) * 1)]
        y_train_set = smiley_set_seniments[0 : int(len(smiley_set_seniments) * 0.98)]
        y_dev_set = smiley_set_seniments[int(len(smiley_set_seniments) * 0.98):int(len(smiley_set_seniments) * 1)]

        print "Length trains_set:", len(train_set)
        print "Length dev_set:", len(dev_set)
        print "Length y_trains_set:", len(y_train_set)
        print "Length y_dev_set:", len(y_dev_set)

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
            randomize=False
        )

        check_freq = train_set_iterator.n_batches/10
        training(nnet_tweets,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_prob_batch,y_dev_set,data_dir=data_dir,parameter_map=parameter_map,early_stop=3,check_freq=check_freq,sup_type='distant')

        batch_number += 1

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
    print nnet_tweets

    params = nnet_tweets.params
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

    n_epochs = 100

    check_freq = train_set_iterator.n_batches/10
    training(nnet_tweets,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_batch,dev_sentiments,data_dir=data_dir,parameter_map=parameter_map,n_outs=3,early_stop=10,check_freq=check_freq,sup_type='supervised')


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
