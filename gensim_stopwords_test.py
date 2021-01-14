import re
import string
import glob

from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim import utils
from gensim.parsing.porter import PorterStemmer

from gensim.corpora.wikicorpus import WikiCorpus
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wiki = WikiCorpus("data/plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2",
                  processes=4, lemmatize=False)


def remove_stopwords(s):
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)

def print_accuracy(model):
    print('Evaluating...\n')
    acc = model.wv.evaluate_word_analogies("questions-words-pl.txt")

    sem_correct = sum((len(acc[1][i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[1][i]['correct']) +
                     len(acc[1][i]['incorrect'])) for i in range(5))
    sem_acc = 100*float(sem_correct)/sem_total

    syn_correct = sum((len(acc[1][i]['correct']) for i in range(5, 14)))
    syn_total = sum((len(acc[1][i]['correct']) +
                     len(acc[1][i]['incorrect'])) for i in range(5, 14))
    syn_acc = 100*float(syn_correct)/syn_total
    print('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))
    print('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return acc


def analogies(model):
    tab = []
    # model.wv.similar_by_word('aktor')
    tab.append(model.wv.most_similar(
        positive=['kobieta', 'król'], negative=['mężczyzna'], topn=3))
    tab.append(model.wv.most_similar(
        positive=['król', 'mężczyzna'], negative=['kobieta'], topn=3))
    tab.append(model.wv.most_similar(
        positive=['mężczyzna', 'kobieta'], negative=['król'], topn=3))
    return tab



file = open("stopwords_pl.txt",'r',encoding='UTF-8')
a=file.readlines()
file.close()

result=" ".join(a).replace("\n","")

STOPWORDS = frozenset(w for w in result.split())

# tekst=remove_stopwords("a to jest jakis tekst to the")


class MySentences(object):
    def __iter__(self):
        for text in wiki.get_texts():
            yield [word for word in text if remove_stopwords(word)]

if __name__ == '__main__':
    sentences = MySentences()
    params = {'vector_size': 300, 'window': 10, 'min_count': 40,
              'workers': 4, 'sample': 1e-3, }
    # word2vec = Word2Vec(sentences, **params)
    

    %time fasttext = FastText(sentences, **params)
    
    analogies(fasttext)
    qw = print_accuracy(fasttext)
