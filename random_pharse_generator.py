from alphabet import Alphabet
import numpy as np
import os
import cPickle
import random

def main():
    HOME_DIR = "semeval_parsed"
    input_fname = '200M'

    data_dir = HOME_DIR + '_' + input_fname

    fname_vocab = os.path.join(data_dir, 'vocab.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)

    n_twwet = 250
    tweet_len = 10
    for line in open('semeval/phrases'):
        print line.replace('\n','')
        line = line.replace('\n','').replace('\r','').split(' ')
        outfile = open('semeval/random_tweet_{}.tsv'.format('_'.join(line)),'w')
        out_lines = []
        for i in xrange(n_twwet):
            tweet = np.random.choice(words,tweet_len,True)
            idx = np.random.choice(xrange(tweet_len - len(line) - 1),2,False)

            #idx = random.randint(0,tweet_len - len(line) - 1)
            #idx = xrange(idx,idx + len(line))
            for k,j in enumerate(idx):
                tweet[j] = line[k]
            out_tweet = ' '.join(tweet.tolist())
            out_line = [str(i),'UNK',out_tweet]
            out_line = '\t'.join(out_line)
            out_lines.append(out_line.encode('utf-8') + '\n')
        outfile.writelines(out_lines)
        outfile.close()

if __name__ == '__main__':
  main()
