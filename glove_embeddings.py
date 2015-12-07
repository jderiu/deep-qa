import numpy as np
import cPickle
import os

from alphabet import Alphabet
from utils import load_glove_vec

def main():
  np.random.seed(123)

  data_dirs = [
              'semeval_parsed'
              ]

  for data_dir in data_dirs:
    fname_vocab = os.path.join(data_dir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)


    for fname,delimiter in [
      ('embeddings/glove.twitter.27B.50d.txt',' '),
      ('embeddings/sswe-u.txt','\t')
                  ]:
      word2vec = load_glove_vec(fname, words,delimiter)


      ndim = len(word2vec[word2vec.keys()[0]])
      print 'ndim', ndim

      random_words_count = 0
      vocab_emb = np.zeros((len(alphabet) + 1, ndim))
      for word, idx in alphabet.iteritems():
        word_vec = word2vec.get(word, None)
        if word_vec is None:
          word_vec = np.random.uniform(-0.25, 0.25, ndim)

          random_words_count += 1
        vocab_emb[idx] = word_vec
      print "Using zero vector as random"
      print 'random_words_count', random_words_count
      print vocab_emb.shape
      outfile = os.path.join(data_dir, 'emb_{}.npy'.format(os.path.basename(fname)))
      print outfile
      np.save(outfile, vocab_emb)


if __name__ == '__main__':
  main()
