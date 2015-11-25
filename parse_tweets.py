import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict


from alphabet import Alphabet

UNKNOWN_WORD_IDX = 0

def load_data(fname):
    tid,tweets = [],[]
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweets.append(tweet.split(" "))

    return tid,tweets

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
    outdir = "semeval"
    train = "semeval/task-B-train-plus-dev.tsv"
    test = "task-B-test2014-twitter.tsv"
    dev = "twitter-test-gold-B.downloaded.dev"

    all_fname = "/tmp/trec-merged.txt"
    files = ' '.join([train, dev, test])

    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)

    tid,tweet = load_data(all_fname)
    print "Number of tweets:",tweet.__len__()

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')

    add_to_vocab(tweet, alphabet)

    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))
    print "alphabet", len(alphabet)

    max_tweet_len = max(map(lambda x: len(x), tweet))
    print "Max tweet lenght:", max_tweet_len

    dummy_word_idx = alphabet.fid

    tweet_idx = convert2indices(tweet, alphabet, dummy_word_idx, max_tweet_len)

    basename, _ = os.path.splitext(os.path.basename(train))
    np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)