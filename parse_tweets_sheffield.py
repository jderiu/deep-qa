import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from alphabet import Alphabet
import gzip
from parse_tweets import preprocess_tweet

emo_dict = {}

def read_emo(path):
    sentiment = 0
    with open(path) as f:
        for line in f:
            splits = line.split(" ")
            emo_dict[splits[0].decode('unicode-escape').encode('latin-1').decode('utf-8')] = float(splits[1])
    print emo_dict

def convertSentiment(tweet):
    sentiment = 0
    tweet = tweet.decode('utf-8')
    for emo,score in emo_dict.iteritems():
        sentiment += score*tweet.count(emo)
        tweet = tweet.replace(emo,"")
    return tweet, sentiment

def load_data(fname):
    tweets,sentiments = [],[]
    tknzr = TweetTokenizer()
    with gzip.open(fname) as f:
        for tweet in f:
            tweet,sentiment = convertSentiment(tweet)
            if sentiment != 0:
                tweets.append(preprocess_tweet(tknzr.tokenize(tweet)))
                sentiments.append(sentiment)
    return tweets,sentiments

if __name__ == '__main__':
    read_emo('emoscores')
    tweets, sentiments = load_data("semeval/smiley_tweets.gz")