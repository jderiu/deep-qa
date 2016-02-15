import numpy
from collections import Counter
from evaluation_metrics import semeval_f1_taskB,semeval_f1_taskD
import os
from nltk import TweetTokenizer
import parse_tweets_sheffield as pts


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


def load_pred(sub_dirs,train_files):
    data_dir = "predictions/taskB"
    X_train = {}
    for sub_dir in sub_dirs:
        spath = os.path.join(data_dir,sub_dir)
        id = 0
        for file in train_files:
            fname = os.path.join(spath,file)
            for line in open(fname,'r'):
                vec = line.split(' ')
                if not id in X_train.keys():
                    X_train[id] = vec
                else:
                    X_train[id].extend(vec)
                id += 1
    return X_train


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



def main():
    test_files = 'semeval/binary/SemEval2016-task4-test.subtask-BD.txt'
    tids,topics,tweets,sentiments = load_data(test_files)

    test_files_pred_jan = 'predictions/taskB/devtestB-2016-opt-2o/prob_predictions/SemEval2016-task4-test.subtask-BD.txt'
    test_files_pred_mau = 'predictions/taskB/predictions_without_test2016/prob_predictions/SemEval2016-task4-test.subtask-BD.txt'

    fjan = open(test_files_pred_jan,'r')
    fmau = open(test_files_pred_mau,'r')

    pred_test_jan = fjan.readlines()
    pred_test_mau = fmau.readlines()

    pred_test_jan = map(lambda x: x.replace('\n',''),pred_test_jan)
    pred_test_mau = map(lambda x: x.replace('\n',''),pred_test_mau)

    pred_test_jan = map(lambda x: x.split(' '),pred_test_jan)
    pred_test_mau = map(lambda x: x.split('\t'),pred_test_mau)

    pred_test_jan = map(lambda x: (float(x[1]),float(x[2])),pred_test_jan)
    pred_test_mau = map(lambda x: (float(x[0]),float(x[1])),pred_test_mau)


    w = 0.65
    w1 = 1-w

    for w in [0.6]:
        w1 = 1-w
        pred_test_avg = []
        for (x0,x1) in zip(pred_test_jan,pred_test_mau):
            p = (x0[1]*w+x1[0]*w1)/2
            n = (x0[0]*w+x1[1]*w1)/2

            if p >= n:
                pred_test_avg.append(1)
            else:
                pred_test_avg.append(0)

        y_pred = numpy.array(pred_test_avg)
        print y_pred

        y_test = numpy.array(sentiments)
        print 'w',w,'B:',semeval_f1_taskB(y_test,y_pred)
        out_data_test = aggregate_topics(topics,y_test)
        out_data_pred= aggregate_topics(topics,y_pred)
        print 'w',w,'D:',semeval_f1_taskD(out_data_test,out_data_pred)

    print Counter(y_pred)

    fout = open('task-B-pred_avg.tsv','w')
    print len(tids)
    print len(topics)
    print y_pred.shape
    for (tid,topic,pred) in zip(tids,topics,y_pred):
        outline = '\t'.join([tid,topic,convertNumber(pred)])
        fout.write(outline+'\n')
    fout.close()


    #y_pred = model.predict(X_test)
    fout_test = open('task-D-gold_avg.tsv','w')
    out_data_test = aggregate_topics(topics,y_test)
    fout_test.writelines(out_data_test)
    fout_test.close()

    fout_pred = open('task-D-pred_avg.tsv','w')
    out_data_pred= aggregate_topics(topics,y_pred)
    fout_pred.writelines(out_data_pred)
    fout_pred.close()
    print Counter(y_pred)

    print semeval_f1_taskD(out_data_test,out_data_pred)


if __name__ == '__main__':
    main()