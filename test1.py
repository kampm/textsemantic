# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:58:37 2020

@author: Kamil
"""

# import pandas as pd

# reader = pd.read_csv("nkjp+wiki-forms-all-100-cbow-hs.txt")

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def ingest():
    data = pd.read_csv('./Sentiment Analysis Dataset.csv',error_bad_lines=False)
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    # data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)    
    return data

data = ingest()
data.head(50)

n=data.shape[0]

def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'
    
def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

# n=100
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Sentiment), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        t=list()
        for a in v:
            t.append(a)
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(t, [label]))
    # print(labelized)
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

x_train[0]
# t=x_train[1]
# print(x_train[0][0])

# for i in x_train[0][0]: 
#     x_train[0][0]=t.append(i)
#     print(i)

n_dim=200
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x[0] for x in tqdm(x_train)])
print([x[0] for x in tqdm(x_train)])
tweet_w2v.train([x[0] for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)

print(tweet_w2v['me'])

print(tweet_w2v.most_similar('good'))

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show

# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
output_file("foo.html")
show(plot_tfidf)

print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
print('building train combines word_vectors with tf-idf ...')
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)
print('train_vecs_w2v shape', train_vecs_w2v.shape)
print('building test combines word_vectors with tf-idf ...')
test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)
print('test_vecs_w2v shape', test_vecs_w2v.shape)


from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,  Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.callbacks import ModelCheckpoint

print('begin to train DNN model for sentiment analysis...')
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=n_dim))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.add(Dense(1, activation='softmax'))
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_vecs_w2v, y_train, epochs=100, batch_size=256, verbose=2)
model.save("modeltest1")
print('Evaluate trained model on test dataset...')
score = model.evaluate(test_vecs_w2v, y_test, batch_size=256, verbose=2)
test_vecs_w2v[:10]
y_test[:10]
print('Accuracy: ', score[1])

loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
from matplotlib import pyplot as plt
plt.plot(epochs, loss, 'bo ', label='Strata trenowania')
# plt.plot(epochs, val_loss, 'b', label='Strata walicacji')
plt.title("str tren i wal")
plt.xlabel('epoki')
plt.ylabel('strata')
plt.legend()
plt.show()

plt.clf()

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo ', label='dokl tren')
plt.plot(epochs, val_acc, 'b', label='dokl wal')
plt.title('dokl trenowania i walidacji')
plt.xlabel('epoki')
plt.ylabel('strata')
plt.legend()
plt.show()
