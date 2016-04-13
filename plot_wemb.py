import os
import matplotlib.pyplot as plt
import pylab as py

pca_after  = open('embeddings/pca_updated_embeddings_distant_L3T85Wcustom.txt','r')
pca_before = open('embeddings/pca_smiley_tweets_embedding_30M.txt','r')

filter_words = ['best','terrible','wtf','love','depressed','sick']
censorship = ['fuck','shit']

data_before = pca_before.readlines()
data_after = pca_after.readlines()

def censor(word):
    if word in censorship:
        return word[0] + '*'*(len(word)-1)
    else:
        return word

data_before = map(lambda x: x.split(' '),data_before)
data_before = filter(lambda x: x[0] not in filter_words,data_before)
data_before = map(lambda x: [censor(x[0]),x[1],x[2]], data_before)

words_before = map(lambda x:x[0],data_before)
x_before = map(lambda x:x[1],data_before)
y_before = map(lambda x:x[2],data_before)

for item in data_before:
    py.scatter(item[1],item[2],s=200*(len(item[0])+5*len(item[0])),marker=r"$ {} $".format(item[0]))

py.xlim(-0.7,0.76)
py.ylim(-0.7,0.7)
py.show()

data_after = map(lambda x: x.split(' '),data_after)
data_after = filter(lambda x: x[0] not in filter_words,data_after)
data_after = map(lambda x: [censor(x[0]),x[1],x[2]], data_after)
words_after = map(lambda x:x[0],data_after)
x_after = map(lambda x:x[1],data_after)
y_after = map(lambda x:x[2],data_after)

for item in data_after:
    py.scatter(item[1],item[2],s=200*(len(item[0])+5*len(item[0])),marker=r"$ {} $".format(item[0]))

py.xlim(-0.7,0.7)
py.ylim(-0.7,0.7)
py.show()