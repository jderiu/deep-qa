import numpy as np
import os
import cPickle

from PIL.Image import alpha_composite

from utils import load_glove_vec
from sklearn.decomposition import TruncatedSVD
from nltk import TweetTokenizer


def main():
    HOME_DIR = "semeval_parsed"
    np.random.seed(123)
    input_fname = '200M'
    embedding = 'custom'
    type = '200M'
    ndim = 52

    data_dir = HOME_DIR + '_' + input_fname
    fname_vocab = os.path.join(data_dir, 'vocab_{}.pickle'.format('topic'))

    tknr = TweetTokenizer()
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    tok_words = {}
    words = []
    for word,idx in alphabet.iteritems():
        tok_word = tknr.tokenize(word.decode('utf-8'))
        tok_words[idx] = tok_word
        words.extend(tok_word)

    print len(tok_words)
    print len(words)
    print "Vocab size", len(alphabet)
    fname,delimiter,ndim = ('embeddings/updated_embeddings_custom_200M'.format(type,str(ndim)),' ',ndim)

    word2vec = load_glove_vec(fname,words,delimiter,ndim)

    print 'len',len(word2vec)
    ndim = len(word2vec[word2vec.keys()[0]])
    print 'ndim', ndim

    random_words_count = 0
    vocab_emb = np.zeros((len(alphabet) + 1, ndim),dtype='float32')

    for idx,tok_word in tok_words.iteritems():
        isrand = 1
        word_vec = np.zeros(ndim)
        for tok in tok_word:
            if tok in word2vec.keys():
                word_vec += word2vec[tok]
                isrand = 0

        if isrand:
          word_vec = np.random.uniform(-0.25, 0.25, ndim)
          random_words_count += 1
        vocab_emb[idx] = word_vec.astype(np.float32)/len(tok_word)
    print "Using zero vector as random"
    print 'random_words_count', random_words_count

    svd = TruncatedSVD(n_components=5)
    vocab_emb = svd.fit_transform(vocab_emb).astype(np.float32)
    print vocab_emb.shape
    fname = 'embeddings/smiley_tweets_embedding_{}'.format('topic')
    outfile = os.path.join(data_dir, 'emb_{}.npy'.format(os.path.basename(fname)))
    print outfile
    np.save(outfile, vocab_emb)

if __name__ == '__main__':
    main()