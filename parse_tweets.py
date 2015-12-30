from alphabet import Alphabet
import os
import numpy as np
import cPickle
import subprocess
from nltk.tokenize import TweetTokenizer
import parse_tweets_sheffield as pts
import sys

UNKNOWN_WORD_IDX = 0


def convertSentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "neutral" : 1,
        "objective-OR-neutral" : 1,
        "objective" :1
    }.get(sentiment,1)


def load_data(fname,alphabet):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            sentiment = convertSentiment(splits[2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = pts.preprocess_tweet(splits[3])
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
    return tid,tweets,sentiments


def add_to_vocab(data, alphabet):
    for sentence in data:
        for token in sentence:
            alphabet.add(token)

CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"

if __name__ == '__main__':

    input_fname = 'small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    outdir = HOME_DIR + '_' + input_fname
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    smiley_pos = 'semeval/smiley_tweets_{}_pos.gz'.format(input_fname)
    smiley_neg = 'semeval/smiley_tweets_{}_neg.gz'.format(input_fname)

    fname_vocab = os.path.join(outdir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_idx = alphabet.fid
    print "alphabet", len(alphabet)
    print 'dummy_word:',dummy_word_idx

    all_fname = "semeval/all_merged.txt"
    files = ' '.join([train, dev, test,test15])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)
    print "Loading SemEval data"
    tid, tweets, sentiments = load_data(all_fname,alphabet)

    print "Loading Smiley Data"
	#save sheffield tweets
    basename, _ = os.path.splitext(os.path.basename('smiley_tweets_pos'.format(input_fname)))
    nTweets = pts.store_file(smiley_pos,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

    basename, _ = os.path.splitext(os.path.basename('smiley_tweets_neg'.format(input_fname)))
    nTweets = pts.store_file(smiley_neg,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

	#save semeval tweets all
    tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)
    print "Number of tweets:", len(tweets)
    basename, _ = os.path.splitext(os.path.basename("all_merged"))
    np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
    np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

	#save semeval tweets seperate
    files = [train,dev,test,test15]
    for fname in files:
        tid, tweets, sentiments = load_data(fname,alphabet)
        print "Number of tweets:",tweets.__len__()

        tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

