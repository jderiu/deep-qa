import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import parse_tweets_sheffield as pts
from alphabet import Alphabet

UNKNOWN_WORD_IDX = 0


def preprocess_tweet(tweet):
    nUpper = 0
    for i,token in enumerate(tweet):
        if token.startswith("@"):
            tweet[i] = "<user>"
        if token.startswith("http"):
            tweet[i] = "<url>"
        if token.startswith("#"):
            tweet[i] = "<hashtag>"
        if token.isupper():
            nUpper += 1
        tweet[i] = tweet[i].lower()
    if nUpper == tweet.__len__:
        tweet.append("<allcaps>")

    return tweet


def convertSentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "neutral" : 1,
        "objective-OR-neutral" : 1,
        "objective" :1
    }.get(sentiment,1)

def load_data(fname):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer()
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            sentiment = convertSentiment(splits[2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweets.append(preprocess_tweet(tknzr.tokenize(tweet)))
                sentiments.append(sentiment)
    return tid,tweets,sentiments


def add_to_vocab(data, alphabet):
    for sentence in data:
        for token in sentence:
            alphabet.add(token)


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=40):
  data_idx = []
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  return data_idx

if __name__ == '__main__':
    outdir = "semeval_parsed"
    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid

    all_fname = "semeval/all-merged.txt"
    files = ' '.join([train, dev, test,test15])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)
    tid, tweets, sentiments = load_data(all_fname)
    tweets_sh, sentiments_sh = pts.load_data("semeval/smiley_tweets.gz")

    add_to_vocab(tweets, alphabet)
    add_to_vocab(tweets_sh,alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))
    print "alphabet", len(alphabet)


    max_tweet_len = max(map(lambda x: len(x), tweets))
    print "Max tweet lenght:", max_tweet_len
    max_tweet_len_sh = max(map(lambda x: len(x), tweets_sh))
    max_tweet_len = max([max_tweet_len,max_tweet_len_sh])

    tweet_idx = convert2indices(tweets_sh, alphabet, dummy_word_idx, max_tweet_len)
    basename, _ = os.path.splitext(os.path.basename("smiley_twets"))
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
    np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments_sh)

    files = [train,dev,test,test15]
    for fname in files:
        tid, tweets, sentiments = load_data(fname)
        print "Number of tweets:",tweets.__len__()

        tweet_idx = convert2indices(tweets, alphabet, dummy_word_idx, max_tweet_len)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

