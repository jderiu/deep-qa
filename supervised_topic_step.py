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
import time
import getopt
from evaluation_metrics import semeval_f1_taskA,semeval_f1_taskB
from sklearn import metrics


def usage():
    print 'python supervised_topic_step.py -i <small,30M> -e <embedding:glove or custom>'


def main():
    ##########
    # LAYERS #
    #########
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    data_dir = HOME_DIR + '_' + input_fname
    numpy_rng = numpy.random.RandomState(123)
    print "Load Parameters"
    parameter_map = cPickle.load(open(data_dir+'/parameters_distant_L2A.p', 'rb'))
    input_shape = parameter_map['inputShape']
    filter_width = parameter_map['filterWidth']
    n_in = parameter_map['n_in']
    st = parameter_map['st']

    fname_wordembeddings = os.path.join(data_dir, 'emb_smiley_tweets_embedding_topic.npy')
    print "Loading word embeddings from", fname_wordembeddings
    vocab_emb_overlap = numpy.load(fname_wordembeddings)
    ndim = vocab_emb_overlap.shape[1]

    ndim = 5
    fname_vocab = os.path.join(data_dir, 'vocab_{}.pickle'.format('topic'))
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_id = alphabet.fid
    vocab_emb_overlap = (numpy_rng.randn(dummy_word_id + 1, ndim) * 0.25).astype(numpy.float32)

    def relu(x):
        return x * (x > 0)

    activation = relu

    tweets = T.imatrix('tweets_train')
    topics = T.imatrix('topics')
    y = T.lvector('y')
    batch_tweets= T.imatrix('batch_x_q')
    batch_topics = T.imatrix('batch_top')
    batch_y = T.lvector('batch_y')

    lookup_table_words = nn_layers.LookupTableFastStatic(
        W=parameter_map['LookupTableFastStaticW'].get_value(),
        pad=filter_width-1
    )

    lookup_table_topic = nn_layers.LookupTableFast(
        W=vocab_emb_overlap,
        pad=filter_width-1
    )

    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words,lookup_table_topic])

    filter_shape = parameter_map['FilterShape' + str(filter_width)]
    filter_shape = (
        filter_shape[0],
        filter_shape[1],
        filter_shape[2],
        filter_shape[3] + ndim
    )

    input_shape = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3] + ndim
    )

    conv_layers = []

    fan_in = numpy.prod(filter_shape[1:])
    fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
    W_bound = numpy.sqrt(1. / fan_in)
    W_data = numpy.asarray(numpy_rng.uniform(low=-W_bound, high=W_bound, size=(filter_shape[0],filter_shape[1],filter_shape[2],ndim)), dtype=theano.config.floatX)

    W_map = parameter_map['Conv2dLayerW' + str(filter_width)].get_value()

    print W_map.shape
    print W_data.shape
    W_data = numpy.concatenate((W_map,W_data),axis=3)

    conv = nn_layers.Conv2dLayer(
        W=theano.shared(W_data, name="W_conv1d", borrow=True),
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    non_linearity = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB' + str(filter_width)],
        b_size=filter_shape[0],
        activation=activation
    )
    shape1 = parameter_map['PoolingShape1']
    pooling = nn_layers.KMaxPoolLayerNative(shape=shape1,ignore_border=True,st=st)

    input_shape2 = parameter_map['input_shape2'+ str(filter_width)]
    filter_shape2 = parameter_map['FilterShape2' + str(filter_width)]

    con2 = nn_layers.Conv2dLayer(
        W=parameter_map['Conv2dLayerW2' + str(filter_width)],
        rng=numpy_rng,
        input_shape=input_shape2,
        filter_shape=filter_shape2
    )

    non_linearity2 = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB2' + str(filter_width)],
        b_size=filter_shape2[0],
        activation=activation
    )

    shape2 = parameter_map['PoolingShape2']
    pooling2 = nn_layers.KMaxPoolLayerNative(shape=shape2,ignore_border=True)

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
        W=parameter_map['LinearLayerW'],
        b=parameter_map['LinearLayerB'],
        rng=numpy_rng,
        n_in=n_in,
        n_out=n_in,
        activation=activation
    )

    n_outs = 2
    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table,
        join_layer,
        flatten_layer,
        hidden_layer,
        classifier
    ])

    inputs_train = [batch_tweets,
                    batch_topics,
                    batch_y]
    givens_train = {tweets: batch_tweets,
                    topics:batch_topics,
                    y: batch_y}

    inputs_pred = [batch_tweets,
                   batch_topics
                   ]
    givens_pred = {tweets:batch_tweets,
                   topics:batch_topics
                   }

    nnet_tweets.set_input((tweets,topics))
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
        givens=givens_train,
    )

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q,batch_topics) for (batch_x_q,batch_topics) in batch_iterator])
        return preds[:batch_iterator.n_samples]


     #######################
    # Supervised Learining#
    ######################
    batch_size = 1000

    training_2016_tids = numpy.load(os.path.join(data_dir, 'task-BD-train-2016.tids.npy'))
    training_2016_tweets = numpy.load(os.path.join(data_dir, 'task-BD-train-2016.tweets.npy'))
    training_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-BD-train-2016.sentiments.npy'))
    training_2016_topics = numpy.load(os.path.join(data_dir, 'task-BD-train-2016.topics.npy'))

    dev_2016_tids = numpy.load(os.path.join(data_dir, 'task-BD-dev-2016.tids.npy'))
    dev_2016_tweets = numpy.load(os.path.join(data_dir, 'task-BD-dev-2016.tweets.npy'))
    dev_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-BD-dev-2016.sentiments.npy'))
    dev_2016_topics = numpy.load(os.path.join(data_dir, 'task-BD-dev-2016.topics.npy'))

    devtest_2016_tids = numpy.load(os.path.join(data_dir, 'task-BD-devtest-2016.tids.npy'))
    devtest_2016_tweets = numpy.load(os.path.join(data_dir, 'task-BD-devtest-2016.tweets.npy'))
    devtest_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-BD-devtest-2016.sentiments.npy'))
    devtest_2016_topics = numpy.load(os.path.join(data_dir, 'task-BD-devtest-2016.topics.npy'))

    test_2016_tids = numpy.load(os.path.join(data_dir, 'task-BD-test2016.tids.npy'))
    test_2016_tweets = numpy.load(os.path.join(data_dir, 'task-BD-test2016.tweets.npy'))
    test_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-BD-test2016.sentiments.npy'))
    test_2016_topics = numpy.load(os.path.join(data_dir, 'task-BD-test2016.topics.npy'))

    training_full_tweets = numpy.concatenate((training_2016_tweets,dev_2016_tweets),axis=0)
    training_full_tweets = numpy.concatenate((training_full_tweets,devtest_2016_tweets),axis=0)

    training_full_sentiments = numpy.concatenate((training_2016_sentiments,dev_2016_sentiments),axis=0)
    training_full_sentiments = numpy.concatenate((training_full_sentiments,devtest_2016_sentiments),axis=0)

    training_full_topics = numpy.concatenate((training_2016_topics,dev_2016_topics),axis=0)
    training_full_topics = numpy.concatenate((training_full_topics,devtest_2016_topics),axis=0)

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [training_full_tweets,training_full_topics,training_full_sentiments],
        batch_size=batch_size,
        randomize=True
    )

    devtest2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [devtest_2016_tweets,devtest_2016_topics],
        batch_size=batch_size,
        randomize=False
    )

    test2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2016_tweets,test_2016_topics],
        batch_size=batch_size,
        randomize=False
    )

    W_emb_list = [w for w in params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    epoch = 0
    n_epochs = 100
    early_stop = 20
    check_freq = 1
    timer_train = time.time()
    no_best_dev_update = 0
    best_dev_acc = -numpy.inf
    num_train_batches = len(train_set_iterator)
    while epoch < n_epochs:
        timer = time.time()
        for i, (tweet,topic,y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
                train_fn(tweet,topic,y_label)

                if i % check_freq == 0 or i == num_train_batches:
                    y_pred_devtest_2016 = predict_batch(test2016_iterator)
                    dev_acc_2016_devtest = semeval_f1_taskB(test_2016_sentiments,y_pred_devtest_2016)

                    if dev_acc_2016_devtest > best_dev_acc:
                        print('devtest 2016 epoch: {} chunk: {} best_chunk_auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc_2016_devtest,best_dev_acc))

                        best_dev_acc = dev_acc_2016_devtest
                        best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                        no_best_dev_update = 0

                        #cPickle.dump(parameter_map, open(data_dir+'/parameters_{}.p'.format('supervised_posneg'), 'wb'))
                        y_pred_test_2016 = predict_batch(test2016_iterator)
                        numpy.save(data_dir+'/predictions_test_2016',y_pred_test_2016)
                        numpy.save(data_dir+'/predictions_devtest2016',y_pred_devtest_2016)


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

    batch_size = input_shape[0]

    inputs_senvec = [batch_tweets,batch_topics]
    givents_senvec = {tweets:batch_tweets,
                      topics:batch_topics
                      }

    output = nnet_tweets.layers[-2].output

    output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

    sets = [
        (dev_2016_tids,dev_2016_topics,dev_2016_tweets,'task-BD-dev-2016'),
        (training_2016_tids,training_2016_topics,training_2016_tweets,'task-BD-train-2016'),
        (devtest_2016_tids,devtest_2016_topics,devtest_2016_tweets,'task-BD-devtest-2016'),
        (test_2016_tids,test_2016_topics,test_2016_tweets,'SemEval2016-task4-test.subtask-BD')
    ]
    for (fids,ftop,fset,name) in sets:
        test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
            numpy_rng,
            [fset,ftop],
            batch_size=batch_size,
            randomize=False
        )

        counter = 0
        fname = open(os.path.join(data_dir,'sentence_vecs_topic/{}.txt'.format(name)), 'w+')
        for i, (tweet,topic) in enumerate(tqdm(test_set_iterator), 1):
            o = output_fn(tweet,topic)
            for vec in o:
                fname.write(fids[counter])
                for el in numpy.nditer(vec):
                    fname.write(" %f" % el)
                fname.write("\n")
                counter+=1
                if counter == test_set_iterator.n_samples:
                    break

    ##############################
    # Get Predictions Probabilites#
    #############################

    batch_size = input_shape[0]

    output = nnet_tweets.layers[-1].p_y_given_x

    output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

    for (fids,ftop,fset,name) in sets:
        test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
            numpy_rng,
            [fset,ftop],
            batch_size=batch_size,
            randomize=False
        )

        counter = 0
        fname = open(os.path.join(data_dir,'prob_predictions_topic/{}.txt'.format(name)), 'w+')
        for i, (tweet,topic) in enumerate(tqdm(test_set_iterator), 1):
            o = output_fn(tweet,topic)
            for vec in o:
                for el in numpy.nditer(vec):
                    fname.write(" %f" % el)
                fname.write("\n")
                counter+=1
                if counter == test_set_iterator.n_samples:
                    break

    ##################
    # Get Predictions#
    ##################




if __name__ == '__main__':
    main()
