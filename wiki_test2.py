# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:23:04 2021

@author: Kamil
"""

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wiki = WikiCorpus("data/plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2",processes=4, lemmatize=False)

# Dla biblioteki fasttext 
# tab=[]
# for text in wiki.get_texts():
#     tab.append(' '.join(text))
    
# with open('demofile2.txt', 'w' ,encoding="utf-8") as file:
#    for line in tab:
#        file.write(line)
#        file.write("\n")

def print_accuracy(model):
    print('Evaluating...\n')
    acc = model.wv.evaluate_word_analogies("questions-words-pl.txt")

    sem_correct = sum((len(acc[1][i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[1][i]['correct']) + len(acc[1][i]['incorrect'])) for i in range(5))
    sem_acc = 100*float(sem_correct)/sem_total
    
    syn_correct = sum((len(acc[1][i]['correct']) for i in range(5, 14)))
    syn_total = sum((len(acc[1][i]['correct']) + len(acc[1][i]['incorrect'])) for i in range(5,14))
    syn_acc = 100*float(syn_correct)/syn_total
    print('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))
    print('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return acc
       
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

word2vec = Word2Vec.load('wiki.word2vec.model')
word2vec.wv.similar_by_word('aktor')
word2vec.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])

qw = print_accuracy(word2vec)
# vocab = word2vec.wv.index_to_key
# word2vec.save('wiki.word2vec.model')

# ================ FT gensim
from gensim.models.fasttext import FastText

print("FastText w/o char n-grams")
%time fasttext = FastText(sentences,max_n=0,**params) 

fasttext = FastText.load('wiki.fasttext-no-n-grams.model')
fasttext.wv.similar_by_word('aktor')
fasttext.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])

qw = print_accuracy(fasttext)
# print(str(fasttext))
# fasttext.save('wiki.fasttext-no-n-grams.model')


print("FastText")
%time fasttext2 = FastText(sentences,**params) 

fasttext2 = FastText.load('wiki.fasttext.model')
fasttext2.wv.similar_by_word('aktor')
fasttext2.wv.most_similar(positive=['kobieta','król'],negative=['mężczyzna'])

qw = print_accuracy(fasttext2)
# qw = fasttext2.wv.evaluate_word_analogies("questions-words-pl.txt")
# print(str(fasttext2))
# fasttext2.save('wiki.fasttext.model')

# FastText(sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100, alpha=0.025,
#          window=5, min_count=5, max_vocab_size=None, word_ngrams=1, sample=0.001,
#          seed=1, workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75, 
#          cbow_mean=1, hashfxn=<built-in function hash>, epochs=5, null_word=0, 
#          min_n=3, max_n=6, sorted_vocab=1, bucket=2000000, trim_rule=None, 
#          batch_words=10000, callbacks=(), max_final_vocab=None)

# ================ FT fasttext
import fasttext
model = fasttext.train_unsupervised("demofile2.txt", model='cbow')
print(model.get_nearest_neighbors('aktor',k=20))
model.get_analogies("questions-words-pl.txt")
# model.save_model("wikifasttextlibcbow.bin")
model2 = fasttext.train_unsupervised("demofile2.txt", model='skipgram')
print(model2.get_nearest_neighbors('aktor',k=20))
# model2.save_model("wikifasttextlibskipgram.bin")

# input             # training file path (required)
# model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
# lr                # learning rate [0.05]
# dim               # size of word vectors [100]
# ws                # size of the context window [5]
# epoch             # number of epochs [5]
# minCount          # minimal number of word occurences [5]
# minn              # min length of char ngram [3]
# maxn              # max length of char ngram [6]
# neg               # number of negatives sampled [5]
# wordNgrams        # max length of word ngram [1]
# loss              # loss function {ns, hs, softmax, ova} [ns]
# bucket            # number of buckets [2000000]
# thread            # number of threads [number of cpus]
# lrUpdateRate      # change the rate of updates for the learning rate [100]
# t                 # sampling threshold [0.0001]
# verbose           # verbose [2]
