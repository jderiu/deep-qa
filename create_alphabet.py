import sys
from nltk.tokenize import TweetTokenizer
from parse_tweets_sheffield import preprocess_tweet,convert_sentiment
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

    def purge_dict(self,input_fname,min_freq=5,emb='custom',ndim=52):
        if emb == 'glove':
            emb_fname,delimiter,ndim = ('embeddings/glove.twitter.27B.50d.txt',' ',50)
        else:
            emb_fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_{}'.format(input_fname,ndim),' ',ndim)

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
    print 'python create_alphabet.py -i <small,30M> -e <embedding:glove or custom> -t <emb type:30M small,200M>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = 'small'
    embedding = 'glove'
    type = 'small'
    ndim = 100

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:t:d:", ["help", "input=","embedding=",'type=','dim='])
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
        elif o in ('-t','type='):
            type = a
        elif o in ('-d','dim='):
            ndim = int(a)
        else:
            assert False, "unhandled option"

    outdir = HOME_DIR + '_' + input_fname


    print embedding
    print input_fname

    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    smiley_pos = 'semeval/smiley_tweets_{}.gz'.format(input_fname)
    train16 = "semeval/task-A-train-2016.tsv"
    dev2016 = "semeval/task-A-dev-2016.tsv"
    devtest2016 = "semeval/task-A-devtest-2016.tsv"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.txt"
    #smiley_neg = 'semeval/smiley_tweets_{}_neg.gz'.format(input_fname)

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid
    tknzr = TweetTokenizer(reduce_len=True)

    fnames = [
        (train,3),
        (dev,3),
        (test,3),
        (test15,3),
        (train16,2),
        (dev2016,2),
        (devtest2016,2),
        (test2016,2)
    ]

    fnames_gz = [smiley_pos]

    counter = 0

    for (fname,pos) in fnames:
        with open(fname,'r ') as f:
            for line in f:
                tweet = line.split('\t')[pos]
                tweet,_ = convert_sentiment(tweet)
                tweet = tknzr.tokenize(preprocess_tweet(tweet))
                for token in tweet:
                    alphabet.add(token)
        print len(alphabet)

    for fname in fnames_gz:
        with gzip.open(fname,'r') as f:
            for tweet in f:
                tweet,_ = convert_sentiment(tweet)
                tweet = tknzr.tokenize(preprocess_tweet(tweet))
                for token in tweet:
                    alphabet.add(token)
                counter += 1
                if (counter % 1000000) == 0:
                    print 'Precessed Tweets:',counter

        print len(alphabet)

    print 'Alphabet before purge:',len(alphabet)
    alphabet.purge_dict(input_fname=type,min_freq=10,emb=embedding,ndim=ndim)
    print 'Alphabet after purge:',len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab_{}_{}.pickle'.format(embedding,ndim)), 'w'))


if __name__ == '__main__':
    main()