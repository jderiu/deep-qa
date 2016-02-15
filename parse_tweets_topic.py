from alphabet import Alphabet
import os
import numpy as np
import cPickle
import subprocess
from nltk.tokenize import TweetTokenizer
import parse_tweets_sheffield as pts
import sys
import getopt


UNKNOWN_WORD_IDX = 0


def convertSentiment(sentiment):
    return {
        "positive": 1,
        "negative": 0,
        "neutral": 2,
        "objective-OR-neutral": 2,
        "objective": 2
    }.get(sentiment,2)


def convertSentimentC(sentiment):
    return {
        '2': 2,
        '1': 1,
        '0': 0,
        '-1':-1,
        '-2':-2
    }.get(sentiment,0)


def get_topic_indices(tweets,topics,topic_alphabet,max_sent_length=140):
    topic_list = []
    for (tweet,topic) in zip(tweets,topics):
        ex = np.zeros(max_sent_length)*topic_alphabet.fid
        for i, token in enumerate(tweet):
            if token in topic:
                ex[i] = topic_alphabet.get(token,0)
        topic_list.append(ex)
    topic_list = np.array(topic_list).astype('int32')
    return topic_list


def load_data(fname,topic_alphabet):
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
                topics.append(topic)
                topic_alphabet.add(topic)

                tweet = pts.preprocess_tweet(tweet)
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
            else:
                n_not_available += 1

    print "Number of not availalbe tweets:", n_not_available
    return tid,topics,tweets,sentiments


def usage():
    print 'python parse_tweets.py -i <small,30M> -e <vocab:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    outdir = HOME_DIR + '_' + input_fname
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ddir = 'semeval/binary'
    train16 = "task-BD-train-2016.tsv"
    dev2016 = "task-BD-dev-2016.tsv"
    devtest2016 = "task-BD-devtest-2016.tsv"
    test2016 = "SemEval2016-task4-test.subtask-BD.txt"

    fname_vocab = os.path.join(outdir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_idx = alphabet.fid
    print "alphabet", len(alphabet)
    print 'dummy_word:',dummy_word_idx

    topic_alphabet = Alphabet(start_feature_id=0)
    topic_alphabet.add('UNKNOWN_TOPIC_IDX')
    dummy_topic_idx = topic_alphabet.fid

    print "Loading Semeval Data"
    #save semeval tweets seperate
    files = [train16,dev2016,devtest2016,test2016]
    for fname in files:
        fname_ext = os.path.join(ddir,fname)
        tid,topics,tweets, sentiments = load_data(fname_ext,topic_alphabet)
        print "Number of tweets:",len(tweets)

        tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)
        topic_idx = get_topic_indices(tweets,topics,topic_alphabet)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)
        np.save(os.path.join(outdir, '{}.topics.npy'.format(basename)), topic_idx)

    cPickle.dump(topic_alphabet, open(os.path.join(outdir, 'vocab_{}.pickle'.format('topic')), 'w'))


if __name__ == '__main__':
    main()