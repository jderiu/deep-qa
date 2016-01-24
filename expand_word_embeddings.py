import sys
from utils import load_glove_vec
import gzip
from nltk import TweetTokenizer,pos_tag,load
from nltk.tag import _pos_tag
from nltk.tag.perceptron import PerceptronTagger
import numpy as np


def main():
    input_fname = 'small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    tknzr = TweetTokenizer()
    tagger = PerceptronTagger()

    fout = ('embeddings/smiley_tweets_embedding_expanded_{}'.format(input_fname))
    fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_{}'.format(input_fname),' ',52)
    word2vec = load_glove_vec(fname,{},delimiter,ndim)

    tagdict = tagger.tagdict
    tagidx = {}
    nRows = len(word2vec)
    nCols = len(tagdict)

    print nRows,':',nCols

    counter = 0
    for tag in tagdict.keys():
        tagidx[tag] = counter
        counter += 1

    exp_wemb = {}
    for word in word2vec.keys():
        exp_wemb[word] = np.zeros(nCols)

    print tagidx

    train = "semeval/task-B-train-plus-dev.tsv.gz"
    test = "semeval/task-B-test2014-twitter.tsv.gz"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv.gz"
    test15 = "semeval/task-B-test2015-twitter.tsv.gz"
    smiley_pos = 'semeval/smiley_tweets_{}.gz'.format(input_fname)

    it = 0
    files = [train,test,dev,test15,smiley_pos]
    for filen in files:
        for tweet in gzip.open(filen,'rb'):
            tweet = tknzr.tokenize(tweet.decode('utf-8'))
            tags = _pos_tag(tweet, None, tagger)
            for (word,tag) in tags:
                if word in exp_wemb.keys() and tag in tagidx.keys():
                    idx = tagidx[tag]
                    exp_wemb[word][idx] = 1
            if (it%10) == 0:
                print 'Progress:',it
            it += 1

    f = open(fout,'wb')
    for word in exp_wemb:
        f.write(word)
        tags = exp_wemb[word]
        for i in np.nditer(tags):
            f.write(' {}'.format(i))
        fname.write("\n")


if __name__ == '__main__':
    main()