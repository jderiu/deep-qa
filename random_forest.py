from sklearn.ensemble import ExtraTreesClassifier
from scipy.sparse import hstack,vstack
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
import numpy
import os
import sys
import math
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import NMF
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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



def load_senvecs(sub_dirs,train_files):
    data_dir = "models"


    X_train = None
    y_train = None
    for sub_dir in sub_dirs:
        spath = os.path.join(data_dir,sub_dir)
        data_sub = None
        label_sub = None
        for file in train_files:
            fname = os.path.join(spath,file)
            data = load_svmlight_file(fname)
            if data_sub == None:
                data_sub = data[0]
                label_sub = data[1]
            else:
                X = data[0]
                y = data[1]

                data_sub = vstack((data_sub,X))
                label_sub = numpy.concatenate((label_sub,y),axis=0)
        if  X_train == None:
            X_train = data_sub
            y_train = label_sub
        else:
            X_train = hstack((X_train,data_sub))

    return X_train.todense(),y_train


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

    files_test = [
             'featuresDevSet.csv',
             'featuresTest2014.csv',
             'featuresTest2015.csv',
             'featuresTest2016.csv',
             ]

    train_files = [
        'featuresData.csv',
        'featuresDev2016.csv',
        'featuresDevTest2016.csv',
        'featuresTrain2016.csv'
    ]

    print 'Load Data'
    sub_dirs = [
        'model1_2014opt',
        'model2_2015opt',
        'model3_2016devtest_opt'
    ]
    X_train,y_train = load_senvecs(sub_dirs,train_files)
    print 'Load Test'
    X_test2015,y_test2015 = load_senvecs(sub_dirs,['featuresTest2015.csv'])
    X_test2014,y_test2014 = load_senvecs(sub_dirs,['featuresTest2014.csv'])
    X_test2016,y_test2016 = load_senvecs(sub_dirs,['featuresTest2016.csv'])

    #model = NMF(n_components=3, init='random', random_state=0)
    model = TruncatedSVD(n_components=3)
    W_train = model.fit_transform(X_train,y=y_train)
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(W_train[:,0],W_train[:,1],W_train[:,2],c=y_train)
    plt.show()
    #for angle in range(0, 1):
    #    ax.view_init(0, angle)
    #    fig.canvas.draw()


    W_test2015 = model.transform(X_test2015)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(W_test2015[:,0],W_test2015[:,1],W_test2015[:,2],c=y_test2015)
    #plt.show()

    W_test2014 = model.transform(X_test2014)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(W_test2014[:,0],W_test2014[:,1],W_test2014[:,2],c=y_test2014)
    #plt.show()


    W_test2016 = model.transform(X_test2016)
    classifiers = [
        KNeighborsClassifier(n_neighbors=200,weights='distance'),
        ExtraTreesClassifier(n_jobs=2,n_estimators=10000, max_depth=None,max_features=3,min_samples_split=1, random_state=0,class_weight={0:1.74733,1:0.6238,2:0.6288},bootstrap=True),
    ]
    names = [
        'KNN',
        #'ExtraTreesClassifier',
    ]



    for (clf,name) in zip(classifiers,names):
        print 'Learn',name
        model = clf.fit(W_train,y_train)
        print 'Predict'
        y_pred2015 = model.predict(W_test2015)
        y_pred2014 = model.predict(W_test2014)
        print "F1 2015:",semeval_f1(y_test2015,y_pred2015)
        print 'F1 2014:',semeval_f1(y_test2014,y_pred2014)
        y_pred2016 =model.predict(W_test2016)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(W_test2016[:,0],W_test2016[:,1],W_test2016[:,2],c=y_pred2016)
        plt.show()



if __name__ == '__main__':
    main()