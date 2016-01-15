from gensim import models
import gzip
import sys
from parse_tweets_sheffield import preprocess_tweet
from nltk import TweetTokenizer
import logging


class MySentences(object):
    def __init__(self, files):
        self.files = files
        self.tknzr = TweetTokenizer()

    def __iter__(self):
       for fname in self.files:
             for tweet in gzip.open(fname,'rb'):
                 tweet = preprocess_tweet(tweet)
                 tweet = self.tknzr.tokenize(tweet.decode('utf-8'))
                 yield filter(lambda word: ' ' not in word, tweet)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_fname = 'small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    train = "semeval/task-B-train-plus-dev.tsv.gz"
    test = "semeval/task-B-test2014-twitter.tsv.gz"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv.gz"
    test15 = "semeval/task-B-test2015-twitter.tsv.gz"
    smiley_pos = 'semeval/smiley_tweets_{}.gz'.format(input_fname)
    #smiley_neg = 'semeval/smiley_tweets_{}_neg.gz'.format(input_fname)
    files = [train,test,dev,test15,smiley_pos]
    sentences = MySentences(files=files)
    model = models.Word2Vec(sentences, size=100, window=5, min_count=10, workers=7,sg=1,sample=1e-5,hs=1)
    model.save_word2vec_format('embeddings/smiley_tweets_embedding_{}'.format(input_fname),binary=False)


if __name__ == '__main__':
    main()