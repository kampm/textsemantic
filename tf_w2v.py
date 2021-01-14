# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:29:17 2021

@author: Kamil
"""
# https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
from tensorflow.keras.layers import Dense, Embedding, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import collections
import tensorflow as tf
from nltk.corpus import conll2000
import logging
from gensim.models import Word2Vec
from nltk.corpus import brown
import nltk
nltk.download()

logging.basicConfig(
    format='%(asctime)s : %(message)s', level=logging.INFO)


sentences = brown.sents()
print(sentences[:5])

emb_dim = 300

w2v = Word2Vec(sentences, vector_size=300, negative=15, epochs=10, workers=4)

word_vectors = w2v.wv
word_vectors.similar_by_word('yes')
word_vectors.most_similar('yes')
word_vectors.most_similar_cosmul('yes')


train = conll2000.tagged_words("train.txt")
test = conll2000.tagged_words("test.txt")
train[:10]


def get_tag_vocabulary(tagged_words):
    tag2id = {}
    for item in tagged_words:
        tag = item[1]
        tag2id.setdefault(tag, len(tag2id))
    return tag2id


word2id = word_vectors.key_to_index
tag2id = get_tag_vocabulary(train)


def get_int_data(tagged_words, word2id, tag2id):
    X, Y = [], []
    unk_count = 0

    for word, tag in tagged_words:
        Y.append(tag2id.get(tag))
        if word in word2id:
            X.append(word2id.get(word))
        else:
            X.append(unk_index)
            unk_count += 1
    print(unk_count/len(tagged_words))
    return np.array(X), np.array(Y)


X_train, Y_train = get_int_data(train, word2id, tag2id)
X_test, Y_test = get_int_data(test, word2id, tag2id)
Y_train,Y_test=to_categorical(Y_train),to_categorical(Y_test)

def add_new_word(new_word, new_vector, new_index, embedding_matrix, word2id):
    embedding_matrix = np.insert(
        embedding_matrix, [new_index], [new_vector], axis=0)
    word2id = {word: (index+1) if index >= new_index else index
               for word, index in word2id.items()}
    word2id[new_word] = new_index
    return embedding_matrix, word2id


unk_index=0
unk_token="UNK"

embedding_matrix=word_vectors.vectors
unk_vector=embedding_matrix.mean(0)
embedding_matrix,word2id=add_new_word(unk_token,unk_vector,unk_index,embedding_matrix,word2id)

# model
vocab_length=len(embedding_matrix)
model=Sequential()

model.add(Embedding(input_dim=vocab_length,
                    output_dim=emb_dim,
                    weights=[embedding_matrix],
                    input_length=1))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation("tanh"))
model.add(Dense(len(tag2id)))
model.add(Activation("softmax"))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="categorical_crossentropy",
                metrics=["accuracy"])
model.summary()

model.fit(X_train,Y_train,batch_size=128,epochs=1,verbose=1)

id2word=sorted(word2id,key=word2id.get)
_,acc=model.evaluate(X_test,Y_test)       

y_pred=model.predict_classes(X_test) 
error_counter=collections.Counter()
for i in range(len(X_test)):
    correct_tag_id=np.argmax(Y_test[i])
    if y_pred[i] != correct_tag_id:
        word=id2word[X_test[i]]
        error_counter[word]+=1

error_counter.most_common(10)
