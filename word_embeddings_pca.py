import gensim
import numpy as np
from sklearn.decomposition import PCA


def main():
    fname = 'embeddings/words.txt'
    words = set()
    f = open(fname,'r')
    for word in f:
        words.add(word.replace('\r\n',''))
    print words
    max_dist = 0
    fnames = ['updated_embeddings_distant_L3T85Wcustom']
    for fname in fnames:
        print 'Load word-embeddings'
        model = gensim.models.Word2Vec.load_word2vec_format('embeddings/'+fname)
        print 'create word-list'
        for i in xrange(max_dist):
            tmp = set()
            for word in words:
                try:
                    s = model.most_similar(word)
                    tmp |= set(map(lambda (x,y):x,s))
                except:
                    continue
            words |= tmp
            print len(words)

        words = set(words)

        it = 0
        X = np.zeros((len(words),52))
        word_dict = {}
        for word in words:
            word_dict[word] = it
            vec = model[word]
            X[it] = np.asarray(vec)
            it += 1

        pca = PCA(copy=True, n_components=2, whiten=False)
        sol = pca.fit_transform(X=X)
        f = open('embeddings/pca_{}.txt'.format(fname),'w')
        for word in words:
            idx = word_dict[word]
            vec = sol[idx]
            f.write(word)
            for el in np.nditer(vec):
                f.write(" %f" % el)
            f.write('\n')

if __name__ == '__main__':
    main()