## Word2Vec

Word2Vec trained with Gensim 4

10/10 plwiki [plwiki-20210101-pages-articles-multistream.xml.bz2](https://dumps.wikimedia.org/plwiki/20210101/plwiki-20210101-pages-articles-multistream.xml.bz2)  
corpus of 1008640 documents with 291075342 positions (total 2349518 articles, 304757007 positions before pruning articles shorter than 50 words)  
Evaluating word analogies for top 300000 words in the model on questions-words-pl.txt  
capital-common-countries: 54.7% (277/506)  
capital-world: 42.9% (5045/11772)  
city-in-state: 78.8% (104/132)  
currency: 2.7% (16/600)  
family: 53.1% (223/420)  
gram1-adjective-to-adverb: 6.2% (50/800)  
gram2-opposite: 19.7% (30/152)  
gram3-comparative: 45.2% (393/870)  
gram4-superlative: 20.6% (104/506)  
gram5-present-participle: 20.2% (142/702)  
gram6-nationality-adjective: 82.9% (1359/1640)  
gram7-past-tense: 17.9% (10/56)  
gram8-plural: 27.6% (368/1332)  
gram9-verb-aspect: 32.1% (18/56)  
Quadruplets with out-of-vocabulary words: 20.5%  
NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"  
Semantic: 5665/13430, Accuracy: 42.18%  
Syntactic: 2474/6114, Accuracy: 40.46%  
Total accuracy: 41.6% (8139/19544)  
**Word2Vec** - [300d (Google Drive)](https://drive.google.com/file/d/1Vm9P5TmCd2PUsQiZNj976F7u1gpZzr4i/view?usp=sharing)

1/10 plwiki [plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2](https://dumps.wikimedia.org/plwiki/20210101/plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2)  
62382 articles, 1158218 unique tokens, 40486171 corpus positions  
Evaluating word analogies for top 300000 words in the model on questions-words-pl.txt  
capital-common-countries: 33.1% (139/420)  
capital-world: 22.0% (1323/6006)  
city-in-state: 36.4% (48/132)  
currency: 2.4% (11/462)  
family: 42.4% (145/342)  
gram1-adjective-to-adverb: 5.8% (34/588)  
gram2-opposite: 13.0% (7/54)  
gram3-comparative: 50.9% (235/462)  
gram4-superlative: 24.3% (51/210)  
gram5-present-participle: 7.6% (16/210)  
gram6-nationality-adjective: 30.7% (504/1640)  
gram8-plural: 28.8% (304/1056)  
gram9-verb-aspect: 20.0% (6/30)  
Quadruplets with out-of-vocabulary words: 52.7%  
NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"  
Semantic: 1666/7362, Accuracy: 22.63%  
Syntactic: 1157/4250, Accuracy: 27.22%  
Total accuracy: 24.3% (2823/11612)  
**Word2Vec** - [300d (Google Drive)](https://drive.google.com/file/d/16Y6rQW3i8bWDe48tOiZi44uzYkUmFjUN/view?usp=sharing)

## FastText

FastText trained with Gensim 4  

1/10 plwiki [plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2](https://dumps.wikimedia.org/plwiki/20210101/plwiki-20210101-pages-articles-multistream1.xml-p1p187037.bz2)  
62382 articles, 1158218 unique tokens, 40486171 corpus positions  
Evaluating word analogies for top 300000 words in the model on questions-words-pl.txt  
capital-common-countries: 33.8% (142/420)  
capital-world: 23.7% (1423/6006)  
city-in-state: 36.4% (48/132)  
currency: 3.2% (15/462)  
family: 40.6% (139/342)  
gram1-adjective-to-adverb: 6.3% (37/588)  
gram2-opposite: 27.8% (15/54)  
gram3-comparative: 50.2% (232/462)  
gram4-superlative: 20.5% (43/210)  
gram5-present-participle: 8.6% (18/210)  
gram6-nationality-adjective: 34.2% (561/1640)  
gram8-plural: 24.6% (260/1056)  
gram9-verb-aspect: 30.0% (9/30)  
Quadruplets with out-of-vocabulary words: 52.7%  
NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"  
Semantic: 1767/7362, Accuracy: 24.00%  
Syntactic: 1175/4250, Accuracy: 27.65%  
Total accuracy: 25.3% (2942/11612)  
**FastText** - [300d (Google Drive)](https://drive.google.com/file/d/1DnjUrgFAGOsn1KnQzFN9WuiDdujpZKml/view?usp=sharing)

