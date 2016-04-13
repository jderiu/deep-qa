import cPickle
import nn_layers
import numpy
from theano import tensor as T
from theano import function
import theano
import os
from tqdm import tqdm
import sgd_trainer
import time
from evaluation_metrics import semeval_f1_taskA
import sys



def main():
    ##########
    # LAYERS #
    #########
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    test_type = ''
    if len(sys.argv) > 1:
        test_type = sys.argv[1]

    data_dir = HOME_DIR + '_' + input_fname
    numpy_rng = numpy.random.RandomState(123)
    print "Load Parameters"
    parameter_map = cPickle.load(open(data_dir+'/parameters_distant_{}.p'.format(test_type), 'rb'))
    input_shape = parameter_map['inputShape']
    filter_width = parameter_map['filterWidth']
    n_in = parameter_map['qLogisticIn']
    k_max = parameter_map['kmax']

    def relu(x):
        return x * (x > 0)

    activation = relu

    tweets = T.imatrix('tweets_train')
    y = T.lvector('y')
    batch_tweets= T.imatrix('batch_x_q')
    batch_y = T.lvector('batch_y')

    lookup_table_words = nn_layers.LookupTableFast(
        W=parameter_map['LookupTableFastStaticW'].get_value(),
        pad=filter_width-1
    )

    filter_shape = parameter_map['FilterShape' + str(filter_width)]

    conv_layers = []

    conv = nn_layers.Conv2dLayer(
        W=parameter_map['Conv2dLayerW' + str(filter_width)],
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    non_linearity = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB' + str(filter_width)],
        b_size=filter_shape[0],
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
        W=parameter_map['LinearLayerW'],
        b=parameter_map['LinearLayerB'],
        rng=numpy_rng,
        n_in=n_in,
        n_out=n_in,
        activation=activation
    )

    n_outs = 3
    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)

    nnet_tweets = nn_layers.FeedForwardNet(layers=[
        lookup_table_words,
        join_layer,
        flatten_layer,
        hidden_layer,
        classifier
    ])

    inputs_train = [batch_tweets,
                    batch_y]
    givens_train = {tweets: batch_tweets,
                    y: batch_y}

    inputs_pred = [batch_tweets,
                   ]
    givens_pred = {tweets:batch_tweets,
                   }

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
        givens=givens_train,
    )

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q[0]) for batch_x_q in batch_iterator])
        return preds[:batch_iterator.n_samples]


    #######
    #Names#
    #######
    test_2016n = 'Test 2016'
    test_2015n = 'Test 2015'
    test_2014n = 'Test 2014'
    test_2013n = 'Test 2013'
    test_2014ljn = 'Test 2014 LiveJournal'
    test_2014srcn = 'Test 2014 Sarcasm'
    test_2013_smsn = 'Test 2013 SMS'
    train_fulln = 'Training Score'

    ep_pred = {}
    ep_pred[test_2016n] = []
    ep_pred[test_2015n] = []
    ep_pred[test_2014n] = []
    ep_pred[test_2013n] = []
    ep_pred[test_2014ljn] = []
    ep_pred[test_2014srcn] = []
    ep_pred[test_2013_smsn] = []
    ep_pred[train_fulln] = []

     #######################
    # Supervised Learining#
    ######################
    batch_size = 1000

    training2013_tids = numpy.load(os.path.join(data_dir, 'task-B-train.20140221.tids.npy'))
    training2013_tweets = numpy.load(os.path.join(data_dir, 'task-B-train.20140221.tweets.npy'))
    training2013_sentiments = numpy.load(os.path.join(data_dir, 'task-B-train.20140221.sentiments.npy'))

    dev_2013_tids = numpy.load(os.path.join(data_dir, 'task-B-dev.20140225.tids.npy'))
    dev_2013_tweets = numpy.load(os.path.join(data_dir, 'task-B-dev.20140225.tweets.npy'))
    dev_2013_sentiments = numpy.load(os.path.join(data_dir, 'task-B-dev.20140225.sentiments.npy'))

    trainingA_2016_tids = numpy.load(os.path.join(data_dir, 'task-A-train-2016.tids.npy'))
    trainingA_2016_tweets = numpy.load(os.path.join(data_dir, 'task-A-train-2016.tweets.npy'))
    trainingA_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-A-train-2016.sentiments.npy'))

    devA_2016_tids = numpy.load(os.path.join(data_dir, 'task-A-dev-2016.tids.npy'))
    devA_2016_tweets = numpy.load(os.path.join(data_dir, 'task-A-dev-2016.tweets.npy'))
    devA_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-A-dev-2016.sentiments.npy'))

    devtestA_2016_tids = numpy.load(os.path.join(data_dir, 'task-A-devtest-2016.tids.npy'))
    devtestA_2016_tweets = numpy.load(os.path.join(data_dir, 'task-A-devtest-2016.tweets.npy'))
    devtestA_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-A-devtest-2016.sentiments.npy'))

    test_2016_tids = numpy.load(os.path.join(data_dir, 'task-A-test2016.tids.npy'))
    test_2016_tweets = numpy.load(os.path.join(data_dir, 'task-A-test2016.tweets.npy'))
    test_2016_sentiments = numpy.load(os.path.join(data_dir, 'task-A-test2016.sentiments.npy'))

    test_2013_tids = numpy.load(os.path.join(data_dir, 'task-B-test2013-twitter.tids.npy'))
    test_2013_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2013-twitter.tweets.npy'))
    test_2013_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2013-twitter.sentiments.npy'))

    test_2014_tids = numpy.load(os.path.join(data_dir, 'task-B-test2014-twitter.tids.npy'))
    test_2014_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2014-twitter.tweets.npy'))
    test_2014_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2014-twitter.sentiments.npy'))

    test_2015_tids = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter.tids.npy'))
    test_2015_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter.tweets.npy'))
    test_2015_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter.sentiments.npy'))

    test_2013_sms_tids = numpy.load(os.path.join(data_dir, 'task-B-test2013-sms.tids.npy'))
    test_2013_sms_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2013-sms.tweets.npy'))
    test_2013_sms_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2013-sms.sentiments.npy'))

    test_2014_livejournal_tids = numpy.load(os.path.join(data_dir, 'task-B-test2014-livejournal.tids.npy'))
    test_2014_livejournal_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2014-livejournal.tweets.npy'))
    test_2014_livejournal_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2014-livejournal.sentiments.npy'))

    test_2014_sarcasm_tids = numpy.load(os.path.join(data_dir, 'task-B-test2014-twittersarcasm.tids.npy'))
    test_2014_sarcasm_tweets = numpy.load(os.path.join(data_dir, 'task-B-test2014-twittersarcasm.tweets.npy'))
    test_2014_sarcasm_sentiments = numpy.load(os.path.join(data_dir, 'task-B-test2014-twittersarcasm.sentiments.npy'))

    rand_tweets_tids = numpy.load(os.path.join(data_dir, 'random_tweet.tids.npy'))
    rand_tweets_tweets = numpy.load(os.path.join(data_dir, 'random_tweet.tweets.npy'))
    rand_tweets_sentiments = numpy.load(os.path.join(data_dir, 'random_tweet.sentiments.npy'))

    training_full_id = numpy.concatenate((training2013_tids,dev_2013_tids),axis=0)
    training_full_id = numpy.concatenate((training_full_id,trainingA_2016_tids),axis=0)
    training_full_id = numpy.concatenate((training_full_id,devA_2016_tids),axis=0)
    training_full_id = numpy.concatenate((training_full_id,devtestA_2016_tids),axis=0)

    training_full_tweets = numpy.concatenate((training2013_tweets,dev_2013_tweets),axis=0)
    training_full_tweets = numpy.concatenate((training_full_tweets,trainingA_2016_tweets),axis=0)
    training_full_tweets = numpy.concatenate((training_full_tweets,devA_2016_tweets),axis=0)
    training_full_tweets = numpy.concatenate((training_full_tweets,devtestA_2016_tweets),axis=0)

    training_full_sentiments = numpy.concatenate((training2013_sentiments,dev_2013_sentiments),axis=0)
    training_full_sentiments = numpy.concatenate((training_full_sentiments,trainingA_2016_sentiments),axis=0)
    training_full_sentiments = numpy.concatenate((training_full_sentiments,devA_2016_sentiments),axis=0)
    training_full_sentiments = numpy.concatenate((training_full_sentiments,devtestA_2016_sentiments),axis=0)

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [training_full_tweets,training_full_sentiments],
        batch_size=batch_size,
        randomize=True
    )

    train_err_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [training_full_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test_2015_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2015_tweets],
        batch_size=batch_size,
        randomize=False
    )

    dev2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [devA_2016_tweets],
        batch_size=batch_size,
        randomize=False
    )

    devtestA2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [devtestA_2016_tweets],
        batch_size=batch_size,
        randomize=False
    )

    train2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [trainingA_2016_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2016_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test2013_itarator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2013_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test_2014_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2014_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test_2014_sarcasm_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2014_sarcasm_tweets],
        batch_size=batch_size,
        randomize=False
    )


    train2013_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [training2013_tweets],
        batch_size=batch_size,
        randomize=False
    )

    dev_2013_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [dev_2013_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test_2013_sms_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2013_sms_tweets],
        batch_size=batch_size,
        randomize=False
    )

    test_2014_livejournal_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [test_2014_livejournal_tweets],
        batch_size=batch_size,
        randomize=False
    )

    W_emb_list = [w for w in params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    epoch = 0
    n_epochs = 50
    early_stop = 50
    check_freq = 4
    timer_train = time.time()
    no_best_dev_update = 0
    best_dev_acc = -numpy.inf
    num_train_batches = len(train_set_iterator)
    saved12 = False
    while epoch < n_epochs:
        timer = time.time()
        for i, (tweet,y_label) in enumerate(tqdm(train_set_iterator,ascii=True), 1):
                train_fn(tweet,y_label)

                if i % check_freq == 0 or i == num_train_batches:
                    y_pred_dev_2015 = predict_batch(test_2015_iterator)
                    #y_pred_train_2013 = predict_batch(train2013_iterator)
                    #y_pred_train_2016 = predict_batch(train2016_iterator)
                    #y_pred_dev2016 = predict_batch(dev2016_iterator)
                    #y_pred_dev2013 = predict_batch(dev_2013_iterator)
                    y_pred_test_2016 = predict_batch(test2016_iterator)
                    y_pred_test_2014 = predict_batch(test_2014_iterator)
                    y_pred_test_2013 = predict_batch(test2013_itarator)
                    y_pred_test_sms_2013 = predict_batch(test_2013_sms_iterator)
                    y_pred_test_livejournal_2014 = predict_batch(test_2014_livejournal_iterator)
                    y_pred_test_sarcasm_2014 = predict_batch(test_2014_sarcasm_iterator)
                    #y_pred_devtest_2016 = predict_batch(devtestA2016_iterator)
                    y_train_score = predict_batch(train_err_iterator)

                    dev_acc_2015 = semeval_f1_taskA(test_2015_sentiments,y_pred_dev_2015)
                    dev_acc_2014 = semeval_f1_taskA(test_2014_sentiments,y_pred_test_2014)
                    dev_acc_2014_lj = semeval_f1_taskA(test_2014_livejournal_sentiments,y_pred_test_livejournal_2014)
                    dev_acc_2014_srcs = semeval_f1_taskA(test_2014_sarcasm_sentiments,y_pred_test_sarcasm_2014)
                    dev_acc_2013 = semeval_f1_taskA(test_2013_sentiments,y_pred_test_2013)
                    dev_acc_2013_sms = semeval_f1_taskA(test_2013_sms_sentiments,y_pred_test_sms_2013)
                    dev_acc_2016 = semeval_f1_taskA(test_2016_sentiments,y_pred_test_2016)
                    dev_acc_train_err = semeval_f1_taskA(training_full_sentiments,y_train_score)

                    ep_pred[test_2016n].append(dev_acc_2016)
                    ep_pred[test_2015n].append(dev_acc_2015)
                    ep_pred[test_2014n].append(dev_acc_2014)
                    ep_pred[test_2013n].append(dev_acc_2013)
                    ep_pred[test_2014ljn].append(dev_acc_2014_lj)
                    ep_pred[test_2014srcn].append(dev_acc_2014_srcs)
                    ep_pred[test_2013_smsn].append(dev_acc_2013_sms)
                    ep_pred[train_fulln].append(dev_acc_train_err)

                    if dev_acc_2016 > best_dev_acc:
                        best_dev_acc = dev_acc_2016
                        best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                        no_best_dev_update = 0

                        print('2016 epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2016))
                        print('2015 epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2015))
                        print('2014 epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2014))
                        print('2013 epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2013))
                        print('2014lj epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2014_lj))
                        print('2014src epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2014_srcs))
                        print('2013sms epoch: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_2013_sms))
                        print('Train Err: {} chunk: {} best_chunk_auc: {:.4f};'.format(epoch, i, dev_acc_train_err))

                if epoch == 15 and not saved12:
                    saved12 = True
                    ep_12_params = [numpy.copy(p.get_value(borrow=True)) for p in params]


        zerout_dummy_word()

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1
        if no_best_dev_update >= early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    cPickle.dump(ep_pred,open(data_dir+'/supervised_results_{}.p'.format(test_type), 'wb'))

    #######################
    # Get Sentence Vectors#
    ######################
    inputs_senvec = [batch_tweets]
    givents_senvec = {tweets:batch_tweets}
    sets = [
            (rand_tweets_tids,rand_tweets_tweets,'random_tweets')
            (test_2016_tids,test_2016_tweets,'SemEval2016-task4-test.subtask-A'),
            (test_2014_tids,test_2014_tweets,'task-B-test2014-twitter'),
            (test_2015_tids,test_2015_tweets,'task-B-test2015-twitter'),
            (test_2013_tids,test_2013_tweets,'task-B-test2013-twitter'),
            (test_2014_livejournal_tids,test_2014_livejournal_tweets,'task-B-test2014-livejournal'),
            (test_2014_sarcasm_tids,test_2014_sarcasm_tweets,'test_2014_sarcasm'),
            (test_2013_sms_tids,test_2013_sms_tweets,'task-B-test2013-sms'),
            (training2013_tids,training2013_tweets,'task-B-train.20140221'),
            (devA_2016_tids,devA_2016_tweets,'task-A-dev-2016'),
            (trainingA_2016_tids,trainingA_2016_tweets,'task-A-train-2016'),
            (devtestA_2016_tids,devtestA_2016_tweets,'task-A-devtest-2016'),
            (dev_2013_tids,dev_2013_tweets,'task-B-dev.20140225'),
            (training_full_id,training_full_tweets,'training_full_set')
        ]

    get_senvec = False
    if get_senvec:
        batch_size = input_shape[0]

        output = nnet_tweets.layers[-2].output

        output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

        for (fids,fset,name) in sets:
            test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
                numpy_rng,
                [fset],
                batch_size=batch_size,
                randomize=False
            )

            counter = 0
            fname = open(os.path.join(data_dir,'sentence-vecs/{}.txt'.format(name)), 'w+')
            for i, tweet in enumerate(tqdm(test_set_iterator), 1):
                o = output_fn(tweet[0])
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
    print 'Store Predictions'

    get_params = [
        ('best_params',best_params),
        ('ep15params',ep_12_params)
    ]

    for pname, gparams in get_params:
        for i, param in enumerate(gparams):
            params[i].set_value(param, borrow=True)
        batch_size = input_shape[0]

        output = nnet_tweets.layers[-1].p_y_given_x

        output_fn = function(inputs=inputs_senvec, outputs=output,givens=givents_senvec)

        for (fids,fset,name) in sets:
            test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
                numpy_rng,
                [fset],
                batch_size=batch_size,
                randomize=False
            )

            opath_prob = os.path.join(data_dir,'{}/{}/predictions_probs'.format(pname,test_type))
            if not os.path.exists(opath_prob):
                os.makedirs(opath_prob)
                print 'Created Path',opath_prob

            opath_pred = os.path.join(data_dir,'{}/{}/predictions_pred'.format(pname,test_type))
            if not os.path.exists(opath_pred):
                os.makedirs(opath_pred)
                print 'Created Path',opath_pred

            counter = 0
            fname_prob = open(os.path.join(opath_prob,'{}.txt'.format(name)), 'w+')
            fname_pred = open(os.path.join(opath_pred,'{}.txt'.format(name)), 'w+')
            for i, tweet in enumerate(tqdm(test_set_iterator), 1):
                o = output_fn(tweet[0])
                for vec in o:
                    #save pred_prob
                    for el in numpy.nditer(vec):
                        fname_prob.write("%f\t" % el)
                    fname_prob.write("\n")
                    #save pred
                    pred = numpy.argmax(vec)
                    sentiments = {
                        0 : 'negative',
                        1 : 'neutral',
                        2 : 'positive'
                    }
                    fname_pred.write('{}\n'.format(sentiments[pred]))

                    counter+=1
                    if counter == test_set_iterator.n_samples:
                        break


if __name__ == '__main__':
    main()
