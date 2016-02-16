# OVERVIEW

This code implements a convolutional neural network for twitter sentiment classification. It is based on convolutional sentence embedding (Aliaksei Severyn et al., 2015).


# DEPENDENCIES

- python 2.7+
- numpy
- [theano](http://deeplearning.net/software/theano/)
- scikit-learn (sklearn)
- pandas
- tqdm
- fish
- numba
- nltk
- gensim

Python packages can be easily installed using the standard tool: pip install <package>

#SETUP
- Place your unsupervised tweets into the semeval/ folder: it sould be called 'smiley_tweets_200M.gz'
- Either run the create_word_embeddings.py code or copy your word embeddings into the embedding/ folder: call it 'smiley_tweets_embedding_final'


# PREPROCESS
- python create_word_embeddings.py 
- python create_alphabet.py 
- python parse_tweets.py 
- python extract_embeddings.py


# TRAIN AND TEST
- THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python distant_supervised_step.py
	- Runs the distant-supervised step, it saves the model in an object called: parameters_distant.p
- THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python supervised_step.py
	- Runs the supervised step, it reads the parameters_distant.p file, which needs to be there.


# REFERENCES

Peter Clark Xuchen Yao, Benjamin Van Durme and Chris Callison-Burch.
Answer extraction as sequence tagging with tree edit distance.
In NAACL, 2013.

Mengqiu Wang, Noah A. Smith, and Teruko Mitaura.
What is the jeopardy model? a quasi- synchronous grammar for qa.
In EMNLP, 2007.

Aliaksei Severyn, Alessandro Moschitt.
Twitter Sentiment Analysis with Deep Convolutional Neural Networks.
SIGIR’15, August 09 - 13, 2015, Santiago, Chile

