from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy
import os
import parse_tweets_sheffield as pts
from nltk import TweetTokenizer
from utils import load_glove_vec
import itertools
from evaluation_metrics import semeval_f1_taskB
import operator
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def load_data(fname):
    tid,topics,tweets,sentiments = [],[],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            sentiment = convertSentiment(splits[2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                topic = pts.preprocess_tweet(splits[1])
                topic_tok = tknzr.tokenize(topic.decode('utf-8'))
                topics.append(splits[1])

                tweet = pts.preprocess_tweet(tweet)
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
            else:
                n_not_available += 1

    print "Number of not availalbe tweets:", n_not_available
    return tid,topics,tweets,sentiments


def adapt_topic(topics):
    topics_new = []
    for topic in topics:
        topics_new.append(topic.replace('@',''))
    return topics_new


def convertNumber(sentiment):
    return {
        1: 'positive',
        0: 'negative',
    }.get(sentiment,'none')


def convertSentiment(sentiment):
    return {
        "positive": 1,
        "negative": 0,
        "neutral": 2,
        "objective-OR-neutral": 2,
        "objective": 2
    }.get(sentiment,0)


def get_topic_indices(tweets,topics,max_sent_length=140):
    topic_list = []
    for (tweet,topic) in zip(tweets,topics):
        ex = numpy.zeros(max_sent_length)
        for i, token in enumerate(tweet):
            if token in topic:
                ex[i] = 1
        topic_list.append(ex)
    topic_list = numpy.array(topic_list).astype('int32')
    return topic_list


def load_senvecs(sub_dirs,train_files):
    data_dir = "predictions/taskB"
    X_train = {}
    for sub_dir in sub_dirs:
        spath = os.path.join(data_dir,sub_dir)
        id = 0
        for file in train_files:
            fname = os.path.join(spath,file)
            for line in open(fname,'r'):
                vec = line.split(' ')[1:]
                if not id in X_train.keys():
                    X_train[id] = vec
                else:
                    X_train[id].extend(vec)
                id += 1

    return X_train


def load_pred(sub_dirs,train_files,start=1):
    data_dir = "predictions/taskB"
    X_train = {}
    for sub_dir in sub_dirs:
        spath = os.path.join(data_dir,sub_dir)
        id = 0
        for file in train_files:
            fname = os.path.join(spath,file)
            for line in open(fname,'r'):
                vec = line.split(' ')[start:]

                if not id in X_train.keys():
                    X_train[id] = vec
                else:
                    X_train[id].extend(vec)
                id += 1
    return X_train

def getX(ids,sen_vecs):
    ndim = len(sen_vecs[sen_vecs.keys()[0]])
    X = numpy.zeros((len(ids), ndim),dtype='float32')
    counter = 0
    for id in ids:
        vec = sen_vecs[id]
        X[counter] = vec
        counter += 1
    return X


def balance_list(tlist):
    y = map(lambda x : x[1],tlist)
    c = Counter(y)
    diff = abs(c[0] - c[1])
    min_lab = min(c.iteritems(),key=operator.itemgetter(1))[0]
    list_min = filter(lambda x : x[1] == min_lab,tlist)
    rand_smpl = [ list_min[i] for i in sorted(numpy.random.choice(len(list_min),diff,replace=True))]
    tlist.extend(rand_smpl)
    numpy.random.shuffle(tlist)
    return tlist


def balance_train_set(topics,X_train,y_train):
    topic_dict = {}
    m = X_train.shape[1]

    for (topic,x,y) in zip(topics,X_train,y_train):
        if topic not in topic_dict.keys():
            topic_dict[topic] = []
        topic_dict[topic].append((x,y))

    ndim = 0
    for topic in topic_dict.keys():
        topic_dict[topic] = balance_list(topic_dict[topic])
        ndim += len(topic_dict[topic])

    X_train = numpy.zeros((ndim,m))
    y_train = numpy.zeros(ndim)

    counter = 0
    for topic in topic_dict.keys():
        for (x,y) in topic_dict[topic]:
            X_train[counter] = x
            y_train[counter] = y
            counter += 1

    return X_train,y_train


def filter_testset(Xt,yt):
    print Counter(yt)
    boolset = []
    it = 0
    for label in numpy.nditer(yt):
        if not label == 2:
            boolset.append(it)
        it += 1

    Xt = Xt[boolset,:]
    yt = yt[boolset]

    print Xt.shape
    print yt.shape
    return Xt,yt


def aggregate_topics(topics,y):
    def f(v):
        denom = float(v[0]) + float(v[1])
        return [str(v[1]/denom),str(v[0]/denom)]

    topic_dict = {}
    for topic in topics:
        topic_dict[topic] = [0,0]

    for (topic,label) in zip(topics,y):
        if label == 0:
            topic_dict[topic][0] += 1
        else:
            topic_dict[topic][1] += 1

    topic_dict = dict(map(lambda (k,v):(k,f(v)), topic_dict.iteritems()))
    out_data = []
    for topic in sorted(topic_dict.keys()):
        val = topic_dict[topic]
        out_data.append('\t'.join([topic,val[0],val[1]]) + '\n')
    return out_data



def topic_vectors(topics,sentence_vecs):
    #return matrix of all topic vectors
    topic_dict = {}

    for (topic,senvec) in zip(topics,sentence_vecs.values()):
        if topic not in topic_dict.keys():
            topic_dict[topic] = []
        topic_dict[topic].append(senvec)

    topic_vecs = []
    ndim = len(sentence_vecs[sentence_vecs.keys()[0]])
    for topic in topic_dict.keys():
        senvecs = topic_dict[topic]
        topic_vec = numpy.zeros(ndim).astype(numpy.float32)
        denom = len(senvecs)
        for senvec in senvecs:
            topic_vec += numpy.array(senvec).astype(numpy.float32)/denom
        topic_dict[topic] = topic_vec

    for topic in topics:
        topic_vec = topic_dict[topic]
        topic_vecs.append(topic_vec)

    return topic_vecs

def main():
    train_files = [
        'semeval/binary/task-BD-train-2016.tsv',
        'semeval/binary/task-BD-dev-2016.tsv'
    ]

    test_files = [
        'semeval/binary/task-BD-devtest-2016.tsv',
    ]

    train_files_pred = [
        'prob_predictions/task-BD-train-2016.txt',
        'prob_predictions/task-BD-dev-2016.txt'
    ]

    test_files_pred = [
        'prob_predictions/task-BD-devtest-2016.txt'
    ]

    files_test_senvec = [
        'sentence-vecs/task-BD-devtest-2016.txt',
    ]

    train_files_senvec = [
        'sentence-vecs/task-BD-train-2016.txt',
        'sentence-vecs/task-BD-dev-2016.txt'
    ]


    sub_dirs = [
        #'jan_test-2013-optimized',
        #'jan_test-2013sms-optimized',
        #'jan_test-2014-optimized',
        #'jan_test-2014livejournal-optimized',
        #'jan_test-2014sarcasm-optimized',
        #'jan_test-2015-optimized'
        #'devtestB-2016-opt',
        #'test2015optfull',
        'devtestB-2016-opt-2o'
    ]

    #task B F1 2016:        83.59
    #Task B F1 2015:        83.6
    #Task B F1 2014:        85.78
    #Task B F1 2014 lj:     79.8
    #Task B F1 2013sms :    81.87
    #Task B F1 2014sarcasm: 58.18

    print 'Load Senvecs'
    senvecs_train = load_senvecs(sub_dirs,train_files_senvec)
    senvecs_test = load_senvecs(sub_dirs,files_test_senvec)

    print 'Load Preds'
    pred_train = load_pred(sub_dirs,train_files_pred)
    pred_test = load_pred(sub_dirs,test_files_pred)

    print 'Load Topics + Sentiments'
    tweets_train = []
    topics_train = []
    y_train = []
    for fnam in train_files:
        _,topics,tweets,sentiments = load_data(fnam)
        tweets_train.extend(tweets)
        topics_train.extend(topics)
        y_train.extend(sentiments)

    tweets_test = []
    topics_test = []
    y_test = []
    id_test = []
    for fnam in test_files:
        tids,topics,tweets,sentiments = load_data(fnam)
        tweets_test.extend(tweets)
        topics_test.extend(topics)
        y_test.extend(sentiments)
        id_test.extend(tids)

    topic_emb_train = topic_vectors(topics_train,senvecs_train)
    topic_emb_test = topic_vectors(topics_test,senvecs_test)

    l = len(senvecs_train)
    ndim = len(senvecs_train[senvecs_train.keys()[0]]) + len(pred_train[pred_train.keys()[0]]) + 200

    X_train = numpy.zeros((l,ndim))
    y_train = numpy.array(y_train)

    counter = 0
    for (sen_vec,pred,topic) in zip(senvecs_train.values(),pred_train.values(),topic_emb_train):
        fe_vec = sen_vec + (pred)
        fe_vec = numpy.array(fe_vec)
        tmp = numpy.concatenate((fe_vec,topic))

        X_train[counter] = tmp
        counter += 1

    print Counter(y_train)
    X_train,y_train = balance_train_set(topics_train,X_train,y_train)

    l = len(senvecs_test)
    X_test = numpy.zeros((l,ndim))
    y_test = numpy.array(y_test)

    counter = 0
    for (sen_vec,pred,topic) in zip(senvecs_test.values(),pred_test.values(),topic_emb_test):
        fe_vec = sen_vec + (pred)
        fe_vec = numpy.array(fe_vec)
        tmp = numpy.concatenate((fe_vec,topic))

        X_test[counter] = tmp
        counter += 1

    X_test,y_test = filter_testset(X_test,y_test)

    model = RandomForestClassifier(n_estimators=15000,max_features=3)
    model.fit(X_train,y_train)

    y_pred = numpy.load('predictions/taskB/devtestB-2016-opt-2o/predictions_devtest_2016.npy')
    print Counter(y_pred)

    fout = open('task-B-pred.tsv','w')
    print len(id_test)
    print len(topics_test)
    print y_pred.shape
    for (tid,topic,pred) in zip(id_test,topics_test,y_pred):
        outline = '\t'.join([tid,topic,convertNumber(pred)])
        fout.write(outline+'\n')
    fout.close()
    print semeval_f1_taskB(y_test,y_pred)

    y_pred = model.predict(X_test)
    fout_test = open('task-D-gold.tsv','w')
    out_data = aggregate_topics(topics_test,y_test)
    fout_test.writelines(out_data)
    fout_test.close()

    fout_pred = open('task-D-pred.tsv','w')
    out_data = aggregate_topics(topics_test,y_pred)
    fout_pred.writelines(out_data)
    fout_pred.close()
    print Counter(y_pred)

if __name__ == '__main__':
    main()