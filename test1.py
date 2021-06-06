# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:58:38 2020

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
LabeledSentence = gensim.models.doc2vec.TaggedDocument # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
emotes = ["BloodTrail","TriHard","demonzW","ResidentSleeper","LUL","demonzZ","cmonBruh","PogChamp","Kappa","demonzLag","DansGame","demonzH","ANELE","demonzGun","BabyRage","xD","EZ","NaM","HYPERCLAP","TriDance","MONKE","D:","Harambe","pepeBASS","demonzX","PEPEDS","TriKool","TriTurbo","Clap","pepeFASTJAM","POLICE","cmonBrug","RapThis","Porvalo","pepeJAM","gorillaGamgam","whotBass","peepoJAMMER","demonzM","PogU","BOOBA","demonzO","Madge","KEKW","AYAYA","BruhW"]

def ingest():
    tweet_train=pd.read_csv('out2.csv',nrows=100000000)
    tweet_train_ban=pd.read_csv('out4.csv')
    tweet_train = tweet_train.dropna()
    data = tweet_train.append(tweet_train_ban)
    # tweets = tweet_train.B.values
    # labels = tweet_train.label.values
    # data = pd.read_csv('out4.csv')
    data = data[["B","label"]]
    data = data.groupby(['B', 'label']).size().reset_index(name='Freq')
    # data = pd.read_csv('./Sentiment Analysis Dataset.csv',error_bad_lines=False)
    # data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    # data = data[data.Sentiment.isnull() == False]
    # # data['Sentiment'] = data['Sentiment'].map(int)
    # data = data[data['SentimentText'].isnull() == False]
    # data.reset_index(inplace=True)
    # data.drop('index', axis=1, inplace=True)
    # print ('dataset loaded with shape', data.shape)    
    return data

for emote in emotes:
    df = df.replace(to_replace =emote, value = '', regex = True)

df = pd.read_csv("out5.csv")
df=df.sort_values(by=['label'], ascending=False)
test=df[:1000]
test.to_csv("out6.csv")
data = ingest()
data.head(50)

n=data.shape[0]



def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        # tokens = filter(lambda t: not t.startswith('@'), tokens)
        # tokens = filter(lambda t: not t.startswith('#'), tokens)
        # tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'
    
def postprocess(data, n=4000000):
    data = data.head(n)
    data['tokens'] = data.B.progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

# n=100
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).label), test_size=0.2)

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
tweet_w2v = Word2Vec(vector_size=n_dim, min_count=0)
tweet_w2v.build_vocab([x[0] for x in tqdm(x_train)])
print([x[0] for x in tqdm(x_train)])
tweet_w2v.train([x[0] for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.epochs)

print(tweet_w2v.wv['me'])

print(tweet_w2v.wv.most_similar('good'))

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
word_vectors = [tweet_w2v.wv[w] for w in list(tweet_w2v.wv.index_to_key)]
# word_vectors = [tweet_w2v.wv[w] for w in list(tweet_w2v.wv.index_to_key)[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.index_to_key)

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
output_file("foo.html")
show(plot_tfidf)

print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=0)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v.wv[word].reshape((1, size)) * tfidf[word]
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
model.add(Dense(64, activation='relu', input_dim=n_dim))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#model.add(Dense(1, activation='softmax'))
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_vecs_w2v, y_train,validation_split=0.2, epochs=100, batch_size=256, verbose=2)
# model.save("modeltest1")
print('Evaluate trained model on test dataset...')
score = model.evaluate(test_vecs_w2v, batch_size=256, verbose=2)
test_vecs_w2v[:10]
y_test[:10]
print('Accuracy: ', score[1])

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

tweet = data.B
sentiment_label = data.label.factorize()
tokenizer = Tokenizer(num_words=500000)
tokenizer.fit_on_texts(tweet)

vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(tweet)

padded_sequence = pad_sequences(encoded_docs, maxlen=64)

embedding_vector_length = 128

model = Sequential()

model.add(Embedding(vocab_size, embedding_vector_length,     
                                     input_length=64) )

model.add(SpatialDropout1D(0.25))

model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', 
                           metrics=['accuracy'])

print(model.summary())
########
top_words=100000
t=Tokenizer(top_words)
t.fit_on_texts(x_train)
x_train=t.texts_to_sequences(x_train)
x_test=t.texts_to_sequences(x_test)

from keras.preprocessing import sequence
max_review_length=300
X_train=sequence.pad_sequences(x_train,maxlen=max_review_length,padding='post')
X_test=sequence.pad_sequences(x_test,maxlen=max_review_length,padding='post')

print(X_train.shape)
print(X_test.shape)

import gensim
word2vec=gensim.models.KeyedVectors.load('wikifull.word2vec.model')

embeding_vector_length=300
embeding_matrix=np.zeros((top_words+1,embeding_vector_length))


for count, value in enumerate(word2vec.wv.index_to_key):
    if count < top_words:
        embeding_vector=word2vec.wv[value]
        # print(embeding_vector)
        embeding_matrix[count]=embeding_vector
        
embeding_matrix
from keras.models import  Sequential
from keras.layers import Embedding,Dropout,Dense,LSTM
model=Sequential()
model.add(Embedding(top_words+1,300,input_length=max_review_length,weights=[embeding_matrix],trainable=False))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs=10,batch_size=100,validation_data=(X_test,y_test))
########
VOCAB_SIZE = 10000

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=10000,
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
########

model = Sequential()
embedding_layer = Embedding(vocab_size, embedding_vector_length, input_length=64 , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(padded_sequence,sentiment_label[0],
                      validation_split=0.2, epochs=5, batch_size=128)



tw = tokenizer.texts_to_sequences([test1])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())

sentiment_label[1][prediction]



import logging

import string
# import cfg
import atexit
import socket
import time
import threading
import random
import socks
import requests
from pprint import pprint
from multiprocessing import Process
from datetime import datetime
from random import randint
from emoji import demojize
from os import listdir
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup


if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    print('Using CPU.')
    device = torch.device("cpu")

torch.load("F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/modelchat.bin")
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s — %(message)s',
                    datefmt='%Y-%m-%d_%H:%M:%S',
                    handlers=[logging.FileHandler('chat.log')])