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

from fse.models import Average, SIF, uSIF
from fse import IndexedList
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors, FastTextKeyedVectors
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

import numpy as np
import re
from nltk import word_tokenize
from fse import CSplitIndexedList
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

file= "CDSCorpus/CDS_train.csv"
similarities, sent_a, sent_b = [], [], []
with open(file, "r",encoding="UTF-8") as f:
    next(f)
    for l in f:
        line = l.rstrip().split("\t")
        similarities.append(float(line[3])/5)
        sent_a.append(line[1])
        sent_b.append(line[2])
similarities = np.array(similarities)
assert len(similarities) == len(sent_a) == len(sent_b)
task_length = len(similarities)

for i, obj in enumerate(zip(similarities, sent_a, sent_b)):
    print(f"{i}\tSim: {obj[0].round(3):.1f}\t{obj[1]:40s}\t{obj[2]:40s}\t")
    if i == 4:
        break

not_punc = re.compile('.*[A-Za-z0-9].*')

def prep_token(token):
    t = token.lower().strip("';.:()").strip('"')
    t = 'not' if t == "n't" else t
    return re.split(r'[-]', t)

def prep_sentence(sentence):
    tokens = []
    for token in word_tokenize(sentence):
        if not_punc.match(token):
            tokens = tokens + prep_token(token)
    return tokens


sentences = CSplitIndexedList(sent_a, sent_b, custom_split=prep_sentence)

sentences[0]
models, results = {}, {}
word2vec = KeyedVectors.load("C:/Users/Kamil/Downloads/word2vec_300_3_polish.bin")


models[f"CBOW-W2V"] = Average(word2vec, lang_freq="pl")
models[f"SIF-W2V"] = SIF(word2vec, components=10)
models[f"uSIF-W2V"] = uSIF(word2vec, length=11)

from gensim.scripts.glove2word2vec import glove2word2vec  
glove = KeyedVectors.load_word2vec_format("C:/Users/Kamil/Downloads/glove_300_3_polish2.txt")
models[f"CBOW-Glove"] = Average(glove,  lang_freq="pl")
print(f"After memmap {sys.getsizeof(glove.vectors)}")
models[f"SIF-Glove"] = SIF(glove, components=15)
models[f"uSIF-Glove"] = uSIF(glove,length=11)

ft = FastTextKeyedVectors.load("C:/Users/Kamil/Downloads/fasttext_100_3_polish.bin")
models[f"CBOW-FT"] = Average(ft, lang_freq="pl")
models[f"SIF-FT"] = SIF(ft, components=10)
models[f"uSIF-FT"] = uSIF(ft, length=11)


s=models[f"uSIF-W2V"]
s.sv[0]

cs, md, ed = [],[],[]
for i, j in zip(range(task_length), range(task_length, 2*task_length)):
    temp1 = s.sv[i].reshape(1, -1)
    temp2 = s.sv[j].reshape(1, -1)
    cs.append((1 - (paired_cosine_distances(temp1, temp2)))[0])
    md.append(-paired_manhattan_distances(temp1, temp2)[0])
    ed.append(-paired_euclidean_distances(temp1, temp2)[0])


eval_pearson_cosine, _ = pearsonr(similarities, cs)
eval_spearman_cosine, _ = spearmanr(similarities, cs)
eval_pearson_manhattan, _ = pearsonr(similarities, md)
eval_spearman_manhattan, _ = spearmanr(similarities, md)
eval_pearson_euclidean, _ = pearsonr(similarities, ed)
eval_spearman_euclidean, _ = spearmanr(similarities, ed)

def compute_similarities(task_length, model):
    sims = []
    for i, j in zip(range(task_length), range(task_length, 2*task_length)):
        sims.append(model.sv.similarity(i,j))
    print(sims)
    return sims

for k, m in models.items():
    m_type  = k.split("-")[0]
    emb_type = k.split("-")[1]
    m.train(sentences)
    r = pearsonr(similarities, compute_similarities(task_length, m))[0].round(4) * 100
    results[f"{m_type}-{emb_type}"] = r
    
    
    print(k, f"{r:2.2f}  m_type{m_type} emb_type{emb_type}")
    
pd.DataFrame.from_dict(results, orient="index", columns=["Pearson"])

# http://mozart.ipipan.waw.pl/~axw/models/lemma/ 300dim
CBOW-W2V    81.12
SIF-W2V     83.55
uSIF-W2V    85.62

# models https://github.com/sdadas/polish-nlp-resources
#100 dim
CBOW-Glove    22.15
SIF-Glove     61.95
uSIF-Glove    62.96
#300 dim
CBOW-Glove    32.46
SIF-Glove     63.03
uSIF-Glove    64.50
#fasttext 100dim
CBOW-FT    47.23
SIF-FT     77.21
uSIF-FT    79.74
#w2v 100dim
CBOW-W2V    36.78
SIF-W2V     63.05
uSIF-W2V    61.06
#w2v 300dim
CBOW-W2V    41.31
SIF-W2V     64.05
uSIF-W2V    63.21
