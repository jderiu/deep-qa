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


def load_data(fname,alphabet,ncols=4):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[ncols - 1]
            sentiment = convertSentiment(splits[ncols - 2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = pts.preprocess_tweet(splits[ncols - 1])
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
    print 'python parse_tweets.py -i <small,30M> -e <vocab:glove or custom>'


def main():
    HOME_DIR = "semeval_parsed"
    input_fname = 'small'
    vocab = 'glove'
    balanced = False
    ndim = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:b,d:", ["help", "input=","embedding=","balanced=","dim="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-e","--embedding"):
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
        elif o in ("-b", "--balanced"):
            balanced = True
        elif o in ("-d", "--dim"):
            ndim = a
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
    train16 = "semeval/task-A-train-2016.tsv"
    dev2016 = "semeval/task-A-dev-2016.tsv"
    devtest2016 = "semeval/task-A-devtest-2016.tsv"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.txt"
    if balanced:
        smiley_tweets = 'semeval/smiley_tweets_{}_balanced.gz'.format(input_fname)
    else:
        smiley_tweets = 'semeval/smiley_tweets_{}.gz'.format(input_fname)

    fname_vocab = os.path.join(outdir, 'vocab_{}_{}.pickle'.format(vocab,ndim))
    alphabet = cPickle.load(open(fname_vocab))
    dummy_word_idx = alphabet.fid
    print "alphabet", len(alphabet)
    print 'dummy_word:',dummy_word_idx

    print "Loading Semeval Data"
    #save semeval tweets seperate
    files = [(train,4),
             (dev,4),
             (test,4),
             (test15,4),
             (train16,3),
             (dev2016,3),
             (devtest2016,3),
             (test2016,3)]
    all_tid,all_tweet,all_sentiment = [],[],[]
    for (fname,ncols) in files:
        tid, tweets, sentiments = load_data(fname,alphabet,ncols=ncols)
        print "Number of tweets:",len(tweets)

        tweet_idx = pts.convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}_{}.tids.npy'.format(basename,vocab)), tid)
        np.save(os.path.join(outdir, '{}_{}.tweets.npy'.format(basename,vocab)), tweet_idx)
        np.save(os.path.join(outdir, '{}_{}.sentiments.npy'.format(basename,vocab)), sentiments)

        all_tid.extend(tid)
        all_tweet.extend(tweets)
        all_sentiment.extend(sentiments)

    #save semeval tweets all
    tweet_idx = pts.convert2indices(all_tweet, alphabet, dummy_word_idx)
    print "Number of tweets:", len(all_tweet)
    basename, _ = os.path.splitext(os.path.basename("all_merged_{}".format(vocab)))
    np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), all_tid)
    np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), all_tweet)
    np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), all_sentiment)


    print "Loading Smiley Data"
    #save sheffield tweets
    #basename, _ = os.path.splitext(os.path.basename('smiley_tweets_pos_{}'.format(vocab)))
    #nTweets = pts.store_file(smiley_pos,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx,sentiment_fname=os.path.join(outdir, '{}.sentiments.npy'.format(basename)))
    #print "Number of tweets:", nTweets

    #basename, _ = os.path.splitext(os.path.basename('smiley_tweets_neg_{}'.format(vocab)))
    #nTweets = pts.store_file(smiley_neg,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx,sentiment_fname=os.path.join(outdir, '{}.sentiments.npy'.format(basename)))
    #print "Number of tweets:", nTweets

    basename, _ = os.path.splitext(os.path.basename('smiley_tweets_{}'.format(vocab)))
    nTweets = pts.store_file(smiley_tweets,os.path.join(outdir, '{}.tweets.npy'.format(basename)),alphabet,dummy_word_idx,sentiment_fname=os.path.join(outdir,'{}.sentiments.npy'.format(basename)))
    print "Number of tweets:", nTweets


if __name__ == '__main__':
    main()