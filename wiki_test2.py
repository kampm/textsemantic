# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:23:04 2021

@author: Kamil
"""

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing

wiki = WikiCorpus("data/plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2",processes=4, lemmatize=False)
# przykladowy 1 artykul z wiki
# for text in wiki.get_texts():
#     print(text)
#     break

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c for c in content], [title])
            
            
documents = TaggedWikiDocument(wiki)

pre = Doc2Vec(min_count=0)
pre.scan_vocab(documents)

for num in range(0, 20):
    print('min_count: {}, size of vocab: '.format(num), pre.prepare_vocab(min_count=num, dry_run=True))


cores = multiprocessing.cpu_count()

models = [
    # PV-DBOW 
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=10, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, min_count=10, epochs =10, workers=cores),
]

models[0].build_vocab(documents)
print(str(models[0]))
models[0].train(documents, total_examples=models[0].corpus_count, epochs=models[0].epochs)


pprint(models[0].docvecs.most_similar(positive=["Piosenka"], topn=20))
# models[0].save(f'wiki{str(models[0])}.model')

# models[1].reset_from(models[0])

# ================ w2v
from gensim.models.word2vec import Word2Vec
# from gensim.models import TfidfModel
# tfidf = TfidfModel(wiki)

class MySentences(object):
    def __iter__(self):
        for text in wiki.get_texts():
            yield [word for word in text]
sentences = MySentences()
params = {'vector_size': 300, 'window': 10, 'min_count': 40, 
          'workers': 4, 'sample': 1e-3,}
word2vec = Word2Vec(sentences, **params)
word2vec.wv.similar_by_word('aktor')

word2vec.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])

# vocab = word2vec.wv.index_to_key
# word2vec.save('wiki.word2vec.model')
word2vec = Word2Vec.load('wiki.word2vec.model')

# ================ FT
from gensim.models.fasttext import FastText
print("FastText w/o char n-grams")
%time fasttext = FastText(sentences,max_n=0,**params) 
fasttext.wv.similar_by_word('aktor')
fasttext.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])

print(str(fasttext))
# fasttext.save('wiki.fasttext-no-n-grams.model')
print("FastText")
%time fasttext2 = FastText(sentences,**params) 
fasttext2.save('wiki.fasttext.model')
fasttext2.wv.similar_by_word('aktor')
fasttext2.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])
