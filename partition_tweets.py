import gzip
from parse_tweets_sheffield import convert_sentiment,read_emo,preprocess_tweet
import shutil
from collections import deque
import sys


def main():
    input_fname = 'small'
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]
        print input_fname

    input_file = 'semeval/smiley_tweets_{}.gz'.format(input_fname)
    output_file = 'semeval/smiley_tweets_{}_balanced.gz'.format(input_fname)
    read_emo('emoscores')

    counter = 0
    pos_counter = 0
    neg_counter = 0
    pos_queue = deque()
    neg_queue = deque()
    f_out = gzip.open(output_file,'w')
    with gzip.open(input_file,'r') as f:
        for tweet in f:
            tweet,sentiment = convert_sentiment(tweet,trim=False)
            tweet = preprocess_tweet(tweet)
            if sentiment == 0:
                pos_queue.append(tweet)
                pos_counter += 1
            if sentiment == 1:
                neg_queue.append(tweet)
                neg_counter += 1
            counter += 1
            while len(neg_queue) > 0 and len(pos_queue) > 0:
                pos_tweet = pos_queue.popleft()
                neg_tweet = neg_queue.popleft()
                f_out.write(pos_tweet)
                f_out.write(neg_tweet)

            if (counter%100000) == 0:
                print "Elements processed:",counter
    print "Pos tweets:",pos_counter
    print "Neg tweets:",neg_counter
    f_out.close()

if __name__ == '__main__':
    main()