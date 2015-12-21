import numpy as np
from nltk.tokenize import TweetTokenizer
import gzip

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
    return tweet, np.sign(sentiment)

UNKNOWN_WORD_IDX = 0


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=140):
  data_idx = []
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  return data_idx


def get_alphabet(fname,alphabet):
    tknzr = TweetTokenizer()
    counter = 0
    max_len = 0
    with gzip.open(fname,'r') as f:
        for tweet in f:
            tweet = preprocess_tweet(tknzr.tokenize(tweet))
            for token in tweet:
                alphabet.add(token)
            max_len = max(max_len,len(tweet))
            counter += 1
            if (counter%100000) == 0:
                print "Elements processed:",counter
    return max_len


def store_file(f_in,f_out,alphabet):
    tknzr = TweetTokenizer()
    counter = 0
    output = open(f_out,'wb')
    batch_size = 600000
    tweet_batch = []
    with gzip.open(f_in,'r') as f:
        for tweet in f:
            tweet = preprocess_tweet(tknzr.tokenize(tweet))
            for token in tweet:
                alphabet.add(token)
            tweet_batch.append(tweet)
            counter += 1
            if counter%batch_size == 0:
                tweet_idx = convert2indices(tweet_batch,alphabet,alphabet.fid)
                np.save(output,tweet_idx)
                print 'Saved tweets:',tweet_idx.shape
                tweet_batch = []
            if (counter%100000) == 0:
                print "Elements processed:",counter

    tweet_idx = convert2indices(tweet_batch,alphabet,alphabet.fid)
    np.save(output,tweet_idx)
    print 'Saved tweets:',tweet_idx.shape
    return counter

def load_data(fname):
    read_emo('emoscores')
    tweets,sentiments = [],[]
    tknzr = TweetTokenizer()
    counter = 0
    with gzip.open(fname,'r') as f:
        for tweet in f:
            tweet,sentiment = convertSentiment(tweet)
            if sentiment != 0:
                tweets.append(preprocess_tweet(tknzr.tokenize(tweet)))
                sentiments.append(sentiment)
            counter += 1
            if (counter%10000) == 0:
                print "Elements processed:",counter
    return tweets,sentiments

if __name__ == '__main__':
    tweets, sentiments = load_data("semeval/smiley_tweets.gz")
    print len(tweets)
