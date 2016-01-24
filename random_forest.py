from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy
import os
import sys
import math


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



def load_senvecs(fname):
    sen_vecs = {}
    with open(fname) as f:
        for line in f:
            splits = line.split(' ')
            id = splits[0]
            sen_vecs[id] = numpy.asarray(splits[1:],dtype='float32')
    return sen_vecs


def getX(ids,sen_vecs):
    ndim = len(sen_vecs[sen_vecs.keys()[0]])
    X = numpy.zeros((len(ids), ndim),dtype='float32')
    counter = 0
    for id in ids:
        vec = sen_vecs[id]
        X[counter] = vec
        counter += 1
    return X


def main():
    input_fname = '200M'
    embedding = 'custom'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    HOME_DIR = "semeval_parsed"
    data_dir = HOME_DIR + '_' + input_fname

    training_tweets = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.tids.npy'.format(embedding)))
    training_sentiments = numpy.load(os.path.join(data_dir, 'task-B-train-plus-dev_{}.sentiments.npy'.format(embedding)))

    dev_tweets1 = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter_{}.tids.npy'.format(embedding)))
    dev_sentiments1 = numpy.load(os.path.join(data_dir, 'task-B-test2015-twitter_{}.sentiments.npy'.format(embedding)))

    dev_tweets2 = numpy.load(os.path.join(data_dir, 'task-B-test2014-twitter_{}.tids.npy'.format(embedding)))
    dev_sentiments2 = numpy.load(os.path.join(data_dir, 'task-B-test2014-twitter_{}.sentiments.npy'.format(embedding)))

    sentence_vectors = load_senvecs(data_dir + '/twitter_sentence_vecs_loaded_f67.txt')
    X_train = getX(training_tweets,sentence_vectors)
    X_dev = getX(dev_tweets1,sentence_vectors)
    X_dev2 = getX(dev_tweets2,sentence_vectors)

    print "Train SVM"
    svm = LinearSVC(C=math.pow(10,3),class_weight={0:1.713,1:0.5593,2:0.72329},multi_class='crammer_singer')
    svm.fit(X_train,training_sentiments)
    print "predict"
    y_pred = svm.predict(X_dev)
    f1 = semeval_f1(dev_sentiments1,y_pred)
    print 'F1 score 2015:',f1
    y_pred = svm.predict(X_dev2)
    f1 = semeval_f1(dev_sentiments2,y_pred)
    print 'F1 score 2014:',f1





if __name__ == '__main__':
    main()