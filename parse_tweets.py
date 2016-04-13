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


def convert_sentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "neutral" : 1,
        "objective-OR-neutral" : 1,
        "objective" :1
    }.get(sentiment,1)


def load_data(fname,alphabet,ncols=4):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[ncols - 1]
            sentiment = convert_sentiment(splits[ncols - 2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = pts.preprocess_tweet(splits[ncols - 1])
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
            else:
                n_not_available += 1

    print "Number of not availalbe tweets:", n_not_available
    return tid,tweets,sentiments


def usage():
    print 'python parse_tweets.py -i <small,30M> -e <vocab:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    outdir = HOME_DIR + '_' + input_fname
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    parse_200M = True
    if len(sys.argv) > 1:
        parse_200M = False

    train2013 = "semeval/task-B-train.20140221.tsv"
    dev2013 = "semeval/task-B-dev.20140225.tsv"
    test2013_sms = "semeval/task-B-test2013-sms.tsv"
    test2013_twitter = "semeval/task-B-test2013-twitter.tsv"
    test2014_twitter = "semeval/task-B-test2014-twitter.tsv"
    test2014_livejournal = "semeval/task-B-test2014-livejournal.tsv"
    test2014_sarcasm = "semeval/test_2014_sarcasm.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    train16 = "semeval/task-A-train-2016.tsv"
    dev2016 = "semeval/task-A-dev-2016.tsv"
    devtest2016 = "semeval/task-A-devtest-2016.tsv"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.tsv"

    rand_tweets = 'semeval/random_tweet_neut.tsv'

    smiley_tweets = 'semeval/smiley_tweets_{}_balanced.gz'.format(input_fname)

    fname_vocab = os.path.join(outdir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_idx = alphabet.fid
    print "alphabet", len(alphabet)
    print 'dummy_word:',dummy_word_idx

    print "Loading Semeval Data"
    #ncol is the number of columns iside the files in semeval
    files = [(train2013,4),
             (dev2013,4),
             (test2013_sms,4),
             (test2013_twitter,4),
             (test2014_twitter,4),
             (test2014_livejournal,4),
             (test2014_sarcasm,4),
             (test15,4),
             (train16,3),
             (dev2016,3),
             (devtest2016,3),
             (test2016,3)]

    files = [(rand_tweets,3)]

    for line in open('semeval/phrases'):
        fname = 'semeval/random_tweet_{}.tsv'.format(line.replace(' ','_').replace('\r','').replace('\n',''))
        ncols = 3
        tid, tweets, sentiments = load_data(fname,alphabet,ncols=ncols)
        print "Number of tweets:",len(tweets)

        tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

    if parse_200M:
        print "Loading Smiley Data"
        basename, _ = os.path.splitext(os.path.basename('smiley_tweets'))
        nTweets = pts.store_file(smiley_tweets,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx,sentiment_fname=os.path.join(outdir,'{}.sentiments.npy'.format(basename)))
        print "Number of tweets:", nTweets


if __name__ == '__main__':
    main()