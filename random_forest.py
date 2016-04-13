import numpy
import os
import parse_tweets_sheffield as pts
from nltk import TweetTokenizer
from sklearn.ensemble import RandomForestClassifier
from evaluation_metrics import semeval_f1_taskA


def load_data(fname,pos):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[pos + 1]
            sentiment = convertSentiment(splits[pos])

            tid.append(splits[0])
            tweet = pts.preprocess_tweet(tweet)
            tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
            tweets.append(tweet_tok)
            sentiments.append(int(sentiment))

    return tid,tweets,sentiments


def convertSentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "neutral": 1,
        "objective-OR-neutral": 2,
        "objective": 2
    }.get(sentiment,0)


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


def load_pred_prob(fname):
    f = open(fname,'r')
    data = f.readlines()
    data = map(lambda x: x.split('\t')[0:3],data)

    return numpy.asarray(data)


def load_pred_pred(fname):
    f = open(fname,'r')
    data = f.readlines()
    data = map(lambda x: convertSentiment(x.replace('\n','')),data)
    out = numpy.asarray(data)
    return out.reshape(out.shape[0],1)


def getX(ids,sen_vecs):
    ndim = len(sen_vecs[sen_vecs.keys()[0]])
    X = numpy.zeros((len(ids), ndim),dtype='float32')
    counter = 0
    for id in ids:
        vec = sen_vecs[id]
        X[counter] = vec
        counter += 1
    return X


def extract_data(data_dir,files,models):
    pred_prob = 'predictions_probs'
    pred_pred = 'predictions_pred'

    pred_model = {}
    X_train = None
    for file in files:
        X_train_file = None
        for model in models:
            ddir = os.path.join(data_dir,model)
            pred_dir = os.path.join(ddir,pred_pred)
            prob_dir = os.path.join(ddir,pred_prob)

            fname_prob = os.path.join(prob_dir,file.replace('tsv','txt'))
            fname_pred = os.path.join(pred_dir,file.replace('tsv','txt'))

            probs = load_pred_prob(fname_prob)
            preds = load_pred_pred(fname_pred)

            prob_pred = numpy.concatenate((probs,preds),axis=1)

            pred_model[model] = preds
            if X_train_file is None:
                X_train_file = prob_pred
            else:
                X_train_file = numpy.concatenate((X_train_file,prob_pred),axis=1)
        if X_train is None:
            X_train = X_train_file
        else:
            X_train = numpy.concatenate((X_train,X_train_file),axis=0)

    return X_train,pred_model


def extract_labels(data_dir,files,sent_pos):
    y = []
    for file,spos in zip(files,sent_pos):
        ddir = os.path.join(data_dir,file)
        _,_,sentiments = load_data(ddir,spos)
        y.extend(sentiments)

    y = numpy.asarray(y)
    return y

def main():
    semeval_dir = 'semeval'

    train2013 = "task-B-train.20140221.tsv"
    dev2013 = "task-B-dev.20140225.tsv"
    test2013_sms = "task-B-test2013-sms.tsv"
    test2013_twitter = "task-B-test2013-twitter.tsv"
    test2014_twitter = "task-B-test2014-twitter.tsv"
    test2014_livejournal = "task-B-test2014-livejournal.tsv"
    test2014_sarcasm = "test_2014_sarcasm.tsv"
    test15 = "task-B-test2015-twitter.tsv"
    train16 = "task-A-train-2016.tsv"
    dev2016 = "task-A-dev-2016.tsv"
    devtest2016 = "task-A-devtest-2016.tsv"
    test2016 = "SemEval2016-task4-test.subtask-A.tsv"

    data_dir = 'ACL/ep15params'
    models = ['L1A','L2A','L2B','L3A','L3B','L3C','L3E','L3F','L3G','L3H']
    pred_prob = 'predictions_probs'
    pred_pred = 'predictions_pred'

    files_training = [train2013,dev2013,train16,dev2016,devtest2016]
    sentpos_train = [2,2,1,1,1]

    files_test = [test2016,test15,test2014_twitter,test2013_twitter,test2014_livejournal,test2014_sarcasm]
    sentpos_test = [1,2,2,2,2,2]

    X_train,_ = extract_data(data_dir,files_training,models)
    y_train = extract_labels(semeval_dir,files_training,sentpos_train)

    testSets = {}
    goldSets = {}
    predModel = {}
    for file,spos in zip(files_test,sentpos_test):
        X_test,y_pred_model = extract_data(data_dir,[file],models)
        y_test = extract_labels(semeval_dir,[file],[spos])

        testSets[file] = X_test
        goldSets[file] = y_test
        predModel[file] = y_pred_model

    print X_train.shape
    print y_train.shape

    print 'Fit model'
    model = RandomForestClassifier(n_estimators=300,max_depth=3,max_features=15,bootstrap=True,n_jobs=4)
    model.fit(X_train,y_train)

    print 'Compute score'
    for file in files_test:
        X_test = testSets[file]
        y_test = goldSets[file]

        y_pred = model.predict(X_test)
        print 'Set:\t{}\tRF\tScore:\t\t{}'.format(file,semeval_f1_taskA(y_test,y_pred))
        for m in models:
            y_pred = predModel[file][m]
            print 'Set:\t{}\t{}\tScore:\t\t{}'.format(file,m,semeval_f1_taskA(y_test,y_pred))

        print '\n'

if __name__ == '__main__':
    main()