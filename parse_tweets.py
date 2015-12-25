import re
import os
import numpy as np
import cPickle
import subprocess
from nltk.tokenize import TweetTokenizer
import parse_tweets_sheffield as pts
from alphabet import Alphabet
import h5py

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
    tknzr = TweetTokenizer(reduce_len=True)
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            sentiment = convertSentiment(splits[2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = pts.preprocess_tweet(splits[3])
                tweets.append(tknzr.tokenize(tweet.decode('utf-8')))
                sentiments.append(int(sentiment))
    return tid,tweets,sentiments


def add_to_vocab(data, alphabet):
    for sentence in data:
        for token in sentence:
            alphabet.add(token)


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=140):
  data_idx = []
  counter = 0
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  counter += 1
  if (counter%10000) == 0:
    print "Number of indexed sentences:",counter
  return data_idx

CL_DIR = "/cluster/work/scr2/jderiu/semeval"
HOME_DIR = "semeval_parsed"

if __name__ == '__main__':
    outdir = HOME_DIR
    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"

    smiley_pos = 'semeval/smiley_tweets_small_pos.gz'
    smiley_neg = 'semeval/smiley_tweets_small_neg.gz'

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid

    all_fname = "semeval/all_merged.txt"
    files = ' '.join([train, dev, test,test15])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)
    print "Loading SemEval data"
    tid, tweets, sentiments = load_data(all_fname)

    print "Done Loading"
    add_to_vocab(tweets, alphabet)

    print "alphabet", len(alphabet)

    print "Loading Smiley Data"
	#save sheffield tweets
    basename, _ = os.path.splitext(os.path.basename("smiley_tweets_pos"))
    nTweets = pts.store_file(smiley_pos,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

    basename, _ = os.path.splitext(os.path.basename("smiley_tweets_neg"))
    nTweets = pts.store_file(smiley_neg,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

    print "alphabet", len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))

	#save semeval tweets all
    tweet_idx = convert2indices(tweets, alphabet, dummy_word_idx)
    print "Number of tweets:", len(tweets)
    basename, _ = os.path.splitext(os.path.basename("all_merged"))
    np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
    np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

	#save semeval tweets seperate
    files = [train,dev,test,test15]
    for fname in files:
        tid, tweets, sentiments = load_data(fname)
        print "Number of tweets:",tweets.__len__()

        tweet_idx = convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

