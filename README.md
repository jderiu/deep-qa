# OVERVIEW

This code implements a convolutional neural network architecture for learning to match question and answer sentences described in the paper:

"Modelling Question-Answer Pairs with Convolutional Neural Networks" submitted to EMNLP, 2015

The network features a state-of-the-art convolutional sentence model, advanced question-answer matching model, and introduces a novel relational model to encode related words in a question-answer pair.

The addressed task is a popular answer sentence selection benchmark, where the goal is for each question to select relevant answer sentences. The dataset was first introduced by (Wang et al., 2007) and further elaborated by (Yao et al., 2013). It is freely [availabe](http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2).

Evaluation is performed using the standard 'trec_eval' script.


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
in the semeval folder place your tweets. 

For the supervised use:
- task-B-test2014-twitter
- task-B-test2015-twitter
- task-B-train-plus-dev
- twitter-test-gold-B.downloaded

For the distant supervised the tweets neet to be gzipped and have the form: smiley_tweets_\<name>_pos.gz and smiley_tweets__\<name>_neg.gz.
If you have all tweets in the same gz you can use the partition_tweets.py \<name> to split the tweets.

In the embeddings folder:
- glove.twitter.27B.50d for the glove embeddings
- or
- smiley_tweets_embedding_\<name> for the custom made embeddings

# PREPROCESS
Note that \<name> is 'small' in the provided sample case.
- python create_word_embeddings.py \<name>
- python create_alphabet.py -i \<name> -e \<embedding: glove or custom>
- python parse_tweets.py -i \<name> -e \<embedding: glove or custom>
- python glove_embeddings.py -i \<name> -e \<embedding: glove or custom>


# TRAIN AND TEST
- THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sentence_vectors.py -i \<name> -e \<embedding: glove or custom>


# REFERENCES

Peter Clark Xuchen Yao, Benjamin Van Durme and Chris Callison-Burch.
Answer extraction as sequence tagging with tree edit distance.
In NAACL, 2013.

Mengqiu Wang, Noah A. Smith, and Teruko Mitaura.
What is the jeopardy model? a quasi- synchronous grammar for qa.
In EMNLP, 2007.
