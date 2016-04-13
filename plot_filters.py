import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors
from alphabet import Alphabet
import os

HOME_DIR = "semeval_parsed"
input_fname = '200M'
data_dir = HOME_DIR + '_' + input_fname

fname_vocab = os.path.join(data_dir, 'vocab_5000.pickle')
alphabet = cPickle.load(open(fname_vocab))
words = alphabet.keys()
print "Vocab size", len(alphabet)
inv_alphabet = {v: k for k, v in alphabet.items()}

print 'Load Params'
best_params = cPickle.load(open(data_dir+'/best_param_supervised_{}.p'.format('L3A'), 'rb'))

filters = best_params[1]
wemb = best_params[0]
print wemb.shape
inv_alphabet[1411779] = 'DUMMY'

model = NearestNeighbors(n_neighbors=3,algorithm='brute',metric='cosine')
model.fit(wemb)

print 'Compute'
n_filters, n_channels, h, w = filters.shape  # n_channels should be == 3

print filters.shape
# Make the panel as square as possible. The next 4 lines are boiler plate, problem can be solved by factorizing n_filters by hand, e.g. 96 = 8x12
side_length = int(np.ceil(np.sqrt(n_filters)))
# pad if necessary
n_filters_ = side_length ** 2
filters_ = np.zeros((n_filters_, n_channels, h, w))
filters_[:n_filters] = filters

counter = 1
fout = open('filter_ords_cosine.txt','w')
# this is the single important line
panel = filters_.reshape(side_length, side_length, n_channels, h, w).transpose(0, 3, 1, 4, 2).reshape(side_length * h, side_length * w).T
#panel = (panel - panel.min()) / panel.ptp()
for i in xrange(side_length):
    for j in xrange(side_length):
        sub_panel = panel[i*w:(i+1)*w,j*h:(j+1)*h].T
        dists,word_idx = model.kneighbors(sub_panel)
        print dists
        for widx in word_idx:
            word0 = inv_alphabet[widx[0]].encode('utf-8')
            word1 = inv_alphabet[widx[1]].encode('utf-8')
            word2 = inv_alphabet[widx[2]].encode('utf-8')
            line = '({},{},{})'.format(word0,word1,word2)
            fout.write(line + ' ')
        fout.write('\n')
        fout.flush()

fout.flush()
fout.close()