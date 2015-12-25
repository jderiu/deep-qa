import gzip
from parse_tweets_sheffield import convertSentiment,read_emo,preprocess_tweet
import shutil
import os
import sys


def main():
    input_fname = 'smiley_tweets_small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]
        print input_fname

    input_file = 'semeval/'+input_fname+'.gz'
    pos_output = open('semeval/'+input_fname+'_pos.txt','w')
    neg_output = open('semeval/'+input_fname+'_neg.txt','w')
    read_emo('emoscores')

    counter = 0
    with gzip.open(input_file,'r') as f:
        for tweet in f:
            tweet,sentiment = convertSentiment(tweet)
            tweet = tweet.encode('utf-8')
            tweet = preprocess_tweet(tweet)
            if sentiment == 1:
                pos_output.write(tweet)
            if sentiment == -1:
                neg_output.write(tweet)
            counter += 1
            if (counter%100000) == 0:
                print "Elements processed:",counter

    with open('semeval/'+input_fname+'_pos.txt','r') as f_in, gzip.open('semeval/'+input_fname+'_pos.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove('semeval/'+input_fname+'_pos.txt')

    with open('semeval/'+input_fname+'_neg.txt','r') as f_in, gzip.open('semeval/'+input_fname+'_neg.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove('semeval/'+input_fname+'_neg.txt')

if __name__ == '__main__':
    main()