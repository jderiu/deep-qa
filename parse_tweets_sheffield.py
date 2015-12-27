import numpy as np
from nltk.tokenize import TweetTokenizer
import gzip
import re
from utils import load_glove_vec

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('@[^\s]+','<user>',tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet

emo_dict = {}
word2vec = load_glove_vec('embeddings/glove.twitter.27B.50d.txt',{},' ')

def read_emo(path):
    with open(path) as f:
        for line in f:
            splits = line.split(" ")
            emo_dict[splits[0].decode('unicode-escape').encode('latin-1').decode('utf-8')] = float(splits[1])
    print emo_dict


def convertSentiment(tweet):
    tweet = tweet.decode('utf-8')
    n_pos = 0
    n_neg = 0
    for emo,score in emo_dict.iteritems():
        if emo in tweet:
            if score == 1:
                n_pos += 1
            elif score == -1:
                n_neg += 1
            tweet = tweet.replace(emo,"")

    if n_pos > 0 and n_neg > 0:
        sentiment = 0
    elif n_pos > 0:
        sentiment = 1
    elif n_neg > 0:
        sentiment = -1
    else:
        sentiment = 0
    return tweet, sentiment

UNKNOWN_WORD_IDX = 0

def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=140):
  data_idx = []
  max_len = 0
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    max_len = max(len(sentence),max_len)
    if len(sentence) > max_sent_length:
        print "Sentence length:",len(sentence)
        print sentence
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  print "Max length in this batch:",max_len
  return data_idx


def normalize_unknown(tweet,word2vec):
    unknown_cnt = 0
    for i,token in enumerate(tweet):
        if word2vec.get(token, None) == None:
            tweet[i] = 'UNK'
            unknown_cnt += 1
    return tweet


def store_file(f_in,f_out,alphabet,dummy_word_idx):
    tknzr = TweetTokenizer(reduce_len=True)
    counter = 0
    output = open(f_out,'wb')
    batch_size = 600000
    tweet_batch = []
    read_emo('emoscores')
    with gzip.open(f_in,'r') as f:
        for tweet in f:
            tweet, _ = convertSentiment(tweet)
            tweet = tweet.encode('utf-8')
            tweet = preprocess_tweet(tweet)
            tweet = tknzr.tokenize(tweet.decode('utf-8'))
            tweet_norm = normalize_unknown(tweet,word2vec)
            for token in tweet_norm:
                alphabet.add(token)
            tweet_batch.append(tweet_norm)
            counter += 1
            if counter%batch_size == 0:
                tweet_idx = convert2indices(tweet_batch,alphabet,dummy_word_idx)
                np.save(output,tweet_idx)
                print 'Saved tweets:',tweet_idx.shape
                tweet_batch = []
            if (counter%100000) == 0:
                print "Elements processed:",counter
    tweet_idx = convert2indices(tweet_batch,alphabet,dummy_word_idx)
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
