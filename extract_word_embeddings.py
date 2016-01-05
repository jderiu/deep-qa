import cPickle
import sys
import os
import getopt
import numpy as np
from alphabet import Alphabet
import re

def usage():
    print 'python extract_word_embeddings.py -i <small,30M> -v <embedding:glove or custom>'


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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

    data_dir = HOME_DIR + '_' + input_fname
    parameter_map = cPickle.load(open(data_dir+'/parameters_f65.p', 'rb'))
    fname_vocab = os.path.join(data_dir, 'vocab_{}.pickle'.format(embedding))
    alphabet = cPickle.load(open(fname_vocab))

    W = parameter_map['LookupTableFastStaticW'].get_value()
    ndim = W.shape[1]
    fname = open('embeddings/updated_embeddings_{}_{}'.format(embedding,input_fname), 'w+')

    counter = 0

    print W.shape
    fname.write(str(W.shape[0])+' '+str(W.shape[1])+'\n')
    for word,idx in alphabet.iteritems():
        word_vec = W[idx]
        word = word.encode('utf-8')
        if 5680 < counter and counter < 5685:
            print word
            print unicode(word)
        try:
            if ' ' not in word:
                fname.write(word)
            else:
                continue
        except:
            continue

        for el in np.nditer(word_vec):
            fname.write(" {}".format(str(el)))
        fname.write("\n")
        counter += 1
        if (counter%100000)== 0:
            print 'Embeddings processed:',counter



#model = gensim.models.Word2Vec.load_word2vec_format('embeddings/updated_embeddings_custom_30M', binary=False,unicode_errors='ignore',encoding='utf-8')
if __name__ == '__main__':
    main()
