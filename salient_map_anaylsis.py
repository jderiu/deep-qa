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


def main(argv):
    ##########
    # LAYERS #
    #########
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'
    data_dir = HOME_DIR + '_' + input_fname

    update_type = 'adadelta'
    batch_size = 1000
    test_type = ''
    use_reg = False
    compute_paramdist = False
    max_norm = 0
    reg = None
    rho=0.95
    eps=1e-6

    argv = map(lambda x: x.replace('\r',''),argv)
    try:
      opts, args = getopt.getopt(argv,"t:u:r:pb:m:e:",["test_type","update=","rho=","batch_size=",'max_norm=','eps='])
    except getopt.GetoptError as e:
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-u", "--update"):
            update_type = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-t", "--testtype"):
            test_type = arg
        elif opt in ("-m", "--max_norm"):
            max_norm = int(arg)
        elif opt in ("-r", "--rho"):
            rho = float(arg)
        elif opt in ("-e", "--eps"):
            eps = float(arg)
        elif opt == "-p":
            compute_paramdist = True

    print update_type
    print batch_size
    print max_norm
    print rho
    print eps


    printeps = abs(int(numpy.floor(numpy.log10(numpy.abs(eps)))))
    print data_dir+'/supervised_results_{}{}{}rho{}eps{}.p'.format(test_type,update_type,max_norm,int(rho*100),printeps)


    numpy_rng = numpy.random.RandomState(123)
    print "Load Parameters"
    parameter_map = cPickle.load(open(data_dir+'/parameters_distant_{}.p'.format(test_type), 'rb'))
    input_shape = parameter_map['inputShape']
    input_shape = (1,input_shape[1],input_shape[2],input_shape[3])
    filter_width = parameter_map['filterWidth']
    n_in = parameter_map['n_in']
    st = parameter_map['st']
    batch_size = input_shape[0]

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

    parameter_map['LookupTableFastStaticW'] = lookup_table_words.W

    filter_shape = parameter_map['FilterShape']

    conv_layers = []

    conv = nn_layers.Conv2dLayer(
        W=parameter_map['Conv2dLayerW'],
        rng=numpy_rng,
        filter_shape=filter_shape,
        input_shape=input_shape
    )

    parameter_map['Conv2dLayerW'] = conv.W

    non_linearity = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB'],
        b_size=filter_shape[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB'] = non_linearity.b

    shape1 = parameter_map['PoolingShape1']
    pooling = nn_layers.KMaxPoolLayerNative(shape=shape1,ignore_border=True,st=st)

    input_shape2 = parameter_map['input_shape2']
    input_shape2 = (1,input_shape2[1],input_shape2[2],input_shape2[3])
    filter_shape2 = parameter_map['FilterShape2']

    con2 = nn_layers.Conv2dLayer(
        W=parameter_map['Conv2dLayerW2' ],
        rng=numpy_rng,
        input_shape=input_shape2,
        filter_shape=filter_shape2
    )

    parameter_map['Conv2dLayerW2'] = con2.W

    non_linearity2 = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB2'],
        b_size=filter_shape2[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB2'] = non_linearity2.b

    shape2 = parameter_map['PoolingShape2']
    st2 = parameter_map['st2']
    pooling2 = nn_layers.KMaxPoolLayerNative(shape=shape2,st=st2,ignore_border=True)

    input_shape3 = parameter_map['input_shape3']
    input_shape3 = (1,input_shape3[1],input_shape3[2],input_shape3[3])
    filter_shape3 = parameter_map['FilterShape3']

    con3 = nn_layers.Conv2dLayer(
        W=parameter_map['Conv2dLayerW3'],
        rng=numpy_rng,
        input_shape=input_shape3,
        filter_shape=filter_shape3
    )

    parameter_map['Conv2dLayerW3'] = con3.W

    non_linearity3 = nn_layers.NonLinearityLayer(
        b=parameter_map['NonLinearityLayerB3'],
        b_size=filter_shape3[0],
        activation=activation
    )

    parameter_map['NonLinearityLayerB3'] = non_linearity3.b

    shape3 = parameter_map['PoolingShape3']
    pooling3 = nn_layers.KMaxPoolLayerNative(shape=shape3,ignore_border=True)

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
    parameter_map['LinearLayerW'] = hidden_layer.W
    parameter_map['LinearLayerB'] = hidden_layer.b

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
    params = nnet_tweets.params
    print nnet_tweets
    print params

    best_params = cPickle.load(open(data_dir+'/best_param_supervised_{}.p'.format(test_type), 'rb'))
    for i, param in enumerate(best_params):
            params[i].set_value(param, borrow=True)

    opath = os.path.join(data_dir,'{}/{}'.format('salient_map',test_type))
    if not os.path.exists(opath):
        os.makedirs(opath)
        print 'Created Path',opath

    sentiments = {
                    0 : 'negative',
                    1 : 'neutral',
                    2 : 'positive'
                }

    for line in open('semeval/phrases'):
        name = 'random_tweet_{}'.format(line.replace(' ','_').replace('\r','').replace('\n',''))
        print name
        test_2016_tids = numpy.load(os.path.join(data_dir, '{}.tids.npy'.format(name)))
        test_2016_tweets = numpy.load(os.path.join(data_dir, '{}.tweets.npy'.format(name)))
        test_2016_sentiments = numpy.load(os.path.join(data_dir, '{}.sentiments.npy'.format(name)))

        for layer in [1]:
            opath = os.path.join(data_dir,'{}/{}/{}/{}'.format('salient_map',test_type,layer,name))
            if not os.path.exists(opath):
                os.makedirs(opath)
                print 'Created Path',opath
            cost = nnet_tweets.layers[-1].training_cost(y)
            for filter_n in [0,30,60,90,120,150,180,199,22,5]:#xrange(200):
                print filter_n
                output = T.sqrt(T.sum(T.grad(cost,params[layer])[filter_n,0,:,:]**2))
                #output = T.grad(cost,params[1])[0,0,:,:].shape
                output_fn = function(inputs=inputs_train,outputs=output,givens=givens_train)

                test_2016_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
                    numpy_rng,
                    [test_2016_tids,test_2016_tweets,test_2016_sentiments],
                    batch_size=batch_size,
                    randomize=False
                )

                magnitudes = []
                for i, (tids,tweet,y_label) in enumerate(tqdm(test_2016_iterator,ascii=True), 1):
                    mag = output_fn(tweet,y_label)
                    for tid,lab,m in zip(numpy.nditer(tids),numpy.nditer(y_label),numpy.nditer(mag)):
                        magnitudes.append((tid,lab,m))

                magnitudes = sorted(magnitudes, key=lambda m: m[2],reverse=True)
                for label in xrange(3):

                    magnitudes_r = filter(lambda x : x[1] == label,magnitudes)
                    magnitudes_r = map(lambda x: (str(x[0]),str(x[1]),str(x[2])),magnitudes_r)
                    magnitudes_r = map(lambda x: '\t'.join(x) + '\n',magnitudes_r)

                    fname_prob = open(os.path.join(opath,'{}_filter{}_{}.txt'.format(name,filter_n,sentiments[label])), 'w+')
                    fname_prob.writelines(magnitudes_r)
                    fname_prob.close()

if __name__ == '__main__':
    main(sys.argv[1:])
