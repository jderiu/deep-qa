import numpy as np
import cPickle
import os
import sys
import getopt
from utils import load_glove_vec
from alphabet import Alphabet


def usage():
    print 'python glove_embeddings.py -i <small,30M> -e <embedding:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    np.random.seed(123)
    input_fname = 'small'
    embedding = 'glove'
    type = 'small'
    ndim = 52

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:t:d:", ["help", "input=","embedding=","type=","dim="])
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
        elif o in ("-t", "--type"):
            type = a
        elif o in ("-d", "--dim"):
            ndim = int(a)
        else:
            assert False, "unhandled option"

    data_dir = HOME_DIR + '_' + input_fname

    fname_vocab = os.path.join(data_dir, 'vocab_{}_{}.pickle'.format(embedding,str(ndim)))
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)
    if embedding == 'glove':
        fname,delimiter,ndim = ('embeddings/glove.twitter.27B.50d.txt',' ',50)
    elif embedding == 'custom':
        fname,delimiter,ndim = ('embeddings/smiley_tweets_embedding_{}_{}'.format(type,str(ndim)),' ',ndim)
    else:
        sys.exit()

    word2vec = load_glove_vec(fname,words,delimiter,ndim)

    ndim = len(word2vec[word2vec.keys()[0]])
    print 'ndim', ndim

    random_words_count = 0
    vocab_emb = np.zeros((len(alphabet) + 1, ndim),dtype='float32')
    for word,idx in alphabet.iteritems():
        word_vec = word2vec.get(word, None)
        if word_vec is None:
          word_vec = np.random.uniform(-0.25, 0.25, ndim)
          random_words_count += 1
        vocab_emb[idx] = word_vec
    print "Using zero vector as random"
    print 'random_words_count', random_words_count
    print vocab_emb.shape
    fname = 'embeddings/smiley_tweets_embedding_{}'.format(input_fname)
    outfile = os.path.join(data_dir, 'emb_{}.npy'.format(os.path.basename(fname)))
    print outfile
    np.save(outfile, vocab_emb)

if __name__ == '__main__':
  main()
