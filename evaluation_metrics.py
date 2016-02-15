import numpy as np
from collections import Counter
import math

def semeval_f1_taskA(y_truth,y_pred):
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


def semeval_f1_taskB(y_truth,y_pred):
    assert len(y_truth) == len(y_pred)

    y_truth = map(lambda x: 1 if x == 2 else x,y_truth)
    y_pred = map(lambda x: 1 if x == 2 else x,y_pred)

    stats = np.zeros((2,2))
    for (t,p) in zip(y_truth,y_pred):
        stats[p][t] += 1

    overall = 0.0
    for label in [0,1]:
        if stats[1][label] + stats[0][label] > 0:
            denomP = stats[1][label] + stats[0][label]
        else:
            denomP = 1

        P = 100.0 * stats[label][label]/denomP

        if stats[label][1] + stats[label][0] > 0:
            denomN = stats[label][1] + stats[label][0]
        else:
            denomN = 1

        N = 100.0*stats[label][label] / denomN

        if P + N > 0:
            denom = P + N
        else:
            denom = 1

        F1 = 2*P*N/denom
        overall += F1

    return overall/2.0


def semeval_mem_taskC(y_truth,y_pred):
    te = Counter(y_truth)
    dist = {}
    for label in te.keys():
        dist[label] = 0.0

    for (t,p) in zip(y_truth,y_pred):
        dist[t] += abs(p-t)

    overall = 0.0
    for label in te.keys():
        classDist = dist[label]/float(te[label])
        overall += classDist

    return overall/5.0


def semeval_f1_taskD(y_truth,y_pred):
    assert len(y_truth) == len(y_pred)
    y_truth = map(lambda x: x.replace('\n','').split('\t'),y_truth)
    y_pred = map(lambda x: x.replace('\n','').split('\t'),y_pred)


    eps = 0.001
    trueStats = {}
    for (topic,pos,neg) in y_truth:
        trueStats[topic] = (float(pos),float(neg))

    propStats = {}
    for (topic,pos,neg) in y_pred:
        propStats[topic] = (float(pos),float(neg))

    for topic in trueStats.keys():
        t = trueStats[topic]
        tp = (t[0] + eps)/(1.0 + eps*2)
        tn = (t[1] + eps)/(1.0 + eps*2)
        trueStats[topic] = (tp,tn)

        p = propStats[topic]
        pp = (p[0] + eps)/(1.0 + eps*2)
        pn = (p[1] + eps)/(1.0 + eps*2)
        propStats[topic] = (pp,pn)

    overall = 0.0
    numTopics = 0

    for topic in trueStats.keys():
        kl = 0.0
        t = trueStats[topic]
        p = propStats[topic]
        posL = t[0]*math.log(t[0]/p[0])
        negL = t[1]*math.log(t[1]/p[1])
        overall += posL + negL
        numTopics += 1
    return overall/numTopics


if __name__ == '__main__':
    y_truth = np.array([2,2,2,0,2,0,0,0,2,2,2,2,0,2,2,2,2,0,2,0,2,0,0,2,2])
    y_pred =  np.array([0,2,2,2,0,0,0,0,2,0,2,0,0,2,2,2,0,0,2,0,2,0,2,2,2])




    print Counter(y_truth)
    print Counter(y_pred)
    print semeval_f1_taskB(y_truth,y_pred)
