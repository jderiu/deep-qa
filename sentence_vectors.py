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
import sys

CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"

def load_smiley_tweets(fname_ps):
    tweet_set = numpy.load(fname_ps)
    while True:
        try:
            batch = numpy.load(fname_ps)
        except:
            break
        tweet_set = numpy.concatenate((tweet_set,batch),axis=0)
    return tweet_set


def training(nnet,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_prob_batch,y_dev_set,parameter_map,n_outs=2,early_stop=5):
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
        for i, (tweet, y_label) in enumerate(tqdm(train_set_iterator), 1):
            train_fn(tweet, y_label)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()

            if i % 3000 == 0 or i == num_train_batches:
                y_pred_dev = predict_prob_batch(dev_set_iterator)
                if n_outs == 2:
                    dev_acc = metrics.roc_auc_score(y_dev_set, y_pred_dev) * 100
                else:
                    dev_acc = metrics.accuracy_score(y_dev_set,y_pred_dev)*100
                if dev_acc > best_dev_acc:
                    print('epoch: {} batch: {} dev auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc,best_dev_acc))
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                    no_best_dev_update = 0
                    cPickle.dump(parameter_map, open('parameters.p', 'wb'))

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


def main():

    input_fname = 'small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]
        print input_fname
    data_dir = HOME_DIR + input_fname

    fname_ps = open(os.path.join(data_dir, 'smiley_tweets_pos.tweets.npy'),'rb')
    smiley_set_tweets_pos = load_smiley_tweets(fname_ps)
    print smiley_set_tweets_pos.shape
    pos_lables = smiley_set_tweets_pos.shape[0]

    fname_neg = open(os.path.join(data_dir, 'smiley_tweets_neg.tweets.npy'),'rb')
    smiley_set_tweets_neg = load_smiley_tweets(fname_neg)
    print smiley_set_tweets_neg.shape
    neg_labels = smiley_set_tweets_neg.shape[0]

    n_tweets = min([pos_lables,neg_labels])
    #[0:n_tweets,:]
    smiley_set_tweets = numpy.concatenate((smiley_set_tweets_pos[0:n_tweets,:],smiley_set_tweets_neg[0:n_tweets,:]),axis=0)
    smiley_set_seniments = numpy.concatenate((numpy.ones(n_tweets),numpy.zeros(n_tweets)),axis=0)

    smiley_set_seniments=smiley_set_seniments.astype(int)
    print smiley_set_tweets.shape

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
    n_epochs = 25
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
    nkernels = 100
    k_max = 1
    num_input_channels = 1
    filter_widths = [3,4,5]
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

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]


    training(nnet_tweets,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_prob_batch,y_dev_set,parameter_map=parameter_map,early_stop=5)

    cPickle.dump(parameter_map, open(data_dir+'/parameters.p', 'wb'))

    #######################
    # Supervised Learining#
    ######################

    training_tweets = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev.tweets.npy'))
    training_sentiments = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev.sentiments.npy'))

    dev_tweets = numpy.load(os.path.join(data_dir, 'twitter-test-gold-B.downloaded.tweets.npy'))
    dev_sentiments = numpy.load(os.path.join(data_dir, 'twitter-test-gold-B.downloaded.sentiments.npy'))

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

    training(nnet_tweets,train_set_iterator,dev_set_iterator,train_fn,n_epochs,predict_batch,dev_sentiments,parameter_map=parameter_map,n_outs=3,early_stop=5)


    #######################
    # Get Sentence Vectors#
    ######################
    test_tweets = numpy.load(os.path.join(data_dir, 'all_merged.tweets.npy'))
    qids_test = numpy.load(os.path.join(data_dir, 'all_merged.tids.npy'))

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
