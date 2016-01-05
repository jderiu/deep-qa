from alphabet import Alphabet
import os
import numpy as np
import cPickle
import subprocess
from nltk.tokenize import TweetTokenizer
import parse_tweets_sheffield as pts
import sys
import getopt

UNKNOWN_WORD_IDX = 0


def convertSentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "neutral" : 1,
        "objective-OR-neutral" : 1,
        "objective" :1
    }.get(sentiment,1)


def load_data(fname,alphabet):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[3]
            sentiment = convertSentiment(splits[2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = pts.preprocess_tweet(splits[3])
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
            else:
                n_not_available += 1

    print "Number of not availalbe tweets:", n_not_available
    return tid,tweets,sentiments


def add_to_vocab(data, alphabet):
    for sentence in data:
        for token in sentence:
            alphabet.add(token)


def usage():
    print 'python parse_tweets.py -i <small,30M> -v <vocab:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = 'small'
    vocab = 'glove'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:v:", ["help", "input=","vocab="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-v","--vocab"):
            if a in ('glove','custom'):
                vocab = a
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
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    smiley_pos = 'semeval/smiley_tweets_{}_pos.gz'.format(input_fname)
    smiley_neg = 'semeval/smiley_tweets_{}_neg.gz'.format(input_fname)

    fname_vocab = os.path.join(outdir, 'vocab_{}.pickle'.format(vocab))
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_idx = alphabet.fid
    print "alphabet", len(alphabet)
    print 'dummy_word:',dummy_word_idx

    all_fname = "semeval/all_merged.txt"
    files = ' '.join([train, dev, test,test15])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)
    print "Loading SemEval data"
    tid, tweets, sentiments = load_data(all_fname,alphabet)
    print "Number of tweets in all_merged",len(tweets)

    print "Loading Smiley Data"
	#save sheffield tweets
    basename, _ = os.path.splitext(os.path.basename('smiley_tweets_pos_{}'.format(vocab)))
    nTweets = pts.store_file(smiley_pos,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

    basename, _ = os.path.splitext(os.path.basename('smiley_tweets_neg_{}'.format(vocab)))
    nTweets = pts.store_file(smiley_neg,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx)
    print "Number of tweets:", nTweets

	#save semeval tweets all
    tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)
    print "Number of tweets:", len(tweets)
    basename, _ = os.path.splitext(os.path.basename("all_merged_{}".format(vocab)))
    np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
    np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

	#save semeval tweets seperate
    files = [train,dev,test,test15]
    for fname in files:
        tid, tweets, sentiments = load_data(fname,alphabet)
        print "Number of tweets:",tweets.__len__()

        tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}_{}.tids.npy'.format(basename,vocab)), tid)
        np.save(os.path.join(outdir, '{}_{}.tweets.npy'.format(basename,vocab)), tweet_idx)
        np.save(os.path.join(outdir, '{}_{}.sentiments.npy'.format(basename,vocab)), sentiments)

if __name__ == '__main__':
    main()