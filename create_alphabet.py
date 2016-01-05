import sys
from nltk.tokenize import TweetTokenizer
from parse_tweets_sheffield import preprocess_tweet,convertSentiment
from utils import load_glove_vec
import gzip
import cPickle
import os
import operator
import getopt

class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id
        self.first = start_feature_id

    def add(self, item):
        idx,freq = self.get(item, (None,None))
        if idx is None:
            idx = self.fid
            self[item] = (idx,1)
            self.fid += 1
        else:
            self[item] = (idx,freq + 1)
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

    def purge_dict(self,input_fname,min_freq=5,emb='glove'):
        if emb == 'glove':
            emb_fname,delimiter,ndim = ('embeddings/glove.twitter.27B.50d.txt',' ',50)
        else:
            emb_fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_{}'.format(input_fname),' ',52)

        word2vec = load_glove_vec(emb_fname,{},delimiter,ndim)
        for k in self.keys():
            idx,freq = self[k]
            if freq < min_freq and word2vec.get(k, None) == None:
                del self[k]
            else:
                self[k] = idx

        self['UNK'] = 0
        counter = self.first
        for k,idx in sorted(self.items(),key=operator.itemgetter(1)):
            self[k] = counter
            counter += 1
        self.fid = counter


def usage():
    print 'python create_alphabet.py -i <small,30M> -e <embedding:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = 'small'
    embedding = 'glove'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:", ["help", "input=","embedding="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-e","--embedding"):
            if a in ('glove','custom'):
                embedding = a
            else:
                usage()
                sys.exit()
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            input_fname = a
        else:
            assert False, "unhandled option"

    outdir = HOME_DIR + '_' + input_fname

    print embedding
    print input_fname

    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    smiley_pos = 'semeval/smiley_tweets_{}_pos.gz'.format(input_fname)
    smiley_neg = 'semeval/smiley_tweets_{}_neg.gz'.format(input_fname)

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid
    tknzr = TweetTokenizer(reduce_len=True)

    fnames = [test,train,dev,test15]
    fnames_gz = [smiley_pos,smiley_neg]

    counter = 0

    for fname in fnames:
        with open(fname,'r ') as f:
            for tweet in f:
                tweet,_ = convertSentiment(tweet)
                tweet = tweet.encode('utf-8')
                tweet = tknzr.tokenize(preprocess_tweet(tweet).decode('utf-8'))
                for token in tweet:
                    alphabet.add(token)
        print len(alphabet)

    for fname in fnames_gz:
        with gzip.open(fname,'r') as f:
            for tweet in f:
                tweet,_ = convertSentiment(tweet)
                tweet = tweet.encode('utf-8')
                tweet = tknzr.tokenize(preprocess_tweet(tweet).decode('utf-8'))
                for token in tweet:
                    alphabet.add(token)
                counter += 1
                if (counter % 100000) == 0:
                    print 'Precessed Tweets:',counter

        print len(alphabet)

    print 'Alphabet before purge:',len(alphabet)
    alphabet.purge_dict(input_fname=input_fname,min_freq=5,emb=embedding)
    print 'Alphabet after purge:',len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab_{}.pickle'.format(embedding)), 'w'))


if __name__ == '__main__':
    main()