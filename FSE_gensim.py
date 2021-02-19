# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:17:52 2021

@author: Kamil
"""
# https://www.aclweb.org/anthology/W18-3012.pdf
# https://github.com/kawine/usif/blob/master/usif.py
# http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
# https://github.com/oborchers/Fast_Sentence_Embeddings
# https://ichi.pro/pl/tworzenie-rekomendacji-dotyczacych-swobodnego-filmu-przy-uzyciu-glebokiego-uczenia-zrob-to-sam-w-mniej-niz-10-minut-42835948738199
# https://arxiv.org/pdf/2005.00630.pdf
# best transformers pl models https://klejbenchmark.com/leaderboard/

from fse.models import uSIF
from fse import IndexedList
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import json
import gensim.downloader as api
import logging
import pandas as pd
from gensim.corpora.wikicorpus import WikiCorpus
logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


czywiesz = pd.read_csv("czywiesz.csv",
    names=["id","question","2","3"],sep=';',error_bad_lines=False, header=None, encoding='utf-8')

sentences = []

# data = api.load("quora-duplicate-questions")
# for d in data:
#     # Let's blow up the data a bit by replicating each sentence.
#     for i in range(8):
#         sentences.append(d["question1"].split())
#         sentences.append(d["question2"].split())

#
for index, row in czywiesz.iterrows():
    sentences.append(row["question"].split())
    
s = IndexedList(sentences)
sentences[255]
print(len(s))

# gensim 3.8.3 works
word2vec = Word2Vec.load('w2v_allwiki_nkjpfull_300.model')
model = uSIF(word2vec, workers=4, lang_freq="pl")
model.train(s)

print(s[500])
model.sv.most_similar(500, indexable=s.items)

model.sv.similar_by_word("teleskop", wv=word2vec, indexable=s.items)

model.sv.similar_by_sentence("Czy łatwo się uczyć".split(), model=model, indexable=s.items)

#######
