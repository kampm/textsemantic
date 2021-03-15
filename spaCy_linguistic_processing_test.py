# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:27:01 2021

@author: Kamil
"""
# Semantyka: Obejmuje badanie znaczenia w języku i może być dalej podzielona na semantykę leksykalną i kompozycyjną.
# - Semantyka leksykalna: Badanie znaczeń słów i symboli za pomocą morfologii i składni.
# - Semantyka kompozycyjna: Badanie relacji między słowami i kombinacji słów oraz rozumienie znaczeń fraz i zdań oraz tego, jak są one powiązane. 
import spacy
import pandas as pd
from spacy import displacy
import textacy # NLP, before and after spaCy
# from spacy.lang.pl.examples import sentences
# Polish pipeline optimized for CPU. Components: tok2vec, morphologizer, tagger, parser, senter, ner, attribute_ruler, lemmatizer.


def display_nlp(doc, include_punct=False):
 """Generate data frame for visualization of spaCy tokens."""
# Text: The original word text.
# Lemma: The base form of the word.
# POS: The simple UPOS part-of-speech tag.
# Tag: The detailed part-of-speech tag.
# Dep: Syntactic dependency, i.e. the relation between tokens.
# Shape: The word shape – capitalization, punctuation, digits.
# is alpha: Is the token an alpha character?
# is stop: Is the token part of a stop list, i.e. the most common words of the language?
 rows = []
 for i, t in enumerate(doc):
     if not t.is_punct or include_punct:
         row = {'token': i, 'text': t.text, 'lemma_':
            t.lemma_,
            'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
            'pos_': t.pos_,'tag_': t.tag_, 'dep_': t.dep_,
            'ent_type_': t.ent_type_, 'ent_iob_':
            t.ent_iob_}
         rows.append(row)
 df = pd.DataFrame(rows).set_index('token')
 df.index.name = None
 return df


def extract_lemmas(doc, **kwargs):
 return [t.lemma_ for t in textacy.extract.words(doc,**kwargs)]


def extract_noun_phrases(doc, preceding_pos=['NOUN'], sep='_'):
 patterns = []
 for pos in preceding_pos:
     patterns.append(f"POS:{pos} POS:NOUN:+")
     spans = textacy.extract.matches(doc, patterns=patterns)
 return [sep.join([t.lemma_ for t in s]) for s in spans]


def extract_entities(doc, include_types=None, sep='_'):
 ents = textacy.extract.entities(doc,
                                 include_types=include_types,
                                 exclude_types=None,
                                 drop_determiners=True,
                                 min_freq=1)
 return [sep.join([t.lemma_ for t in e])+'/'+e.label_ for e in ents]

def extract_nlp(doc):
 return {
     'lemmas': extract_lemmas(doc,
                              exclude_pos=['PART',
                                           'PUNCT',
                                           'DET', 'PRON', 'SYM',
                                           'SPACE'],
                              filter_stops=False),
     'adjs_verbs': extract_lemmas(doc, include_pos=['ADJ',
                                                    'VERB']),
     'nouns': extract_lemmas(doc, include_pos=['NOUN',
                                               'PROPN']),
     'noun_phrases': extract_noun_phrases(doc, ['NOUN']),
     'adj_noun_phrases': extract_noun_phrases(doc, ['ADJ']),
     'entities': extract_entities(doc, ['PERSON', 'ORG',
                                        'GPE', 'LOC'])
 }

# https://spacy.io/models/pl
# 3.0.0 / pl_core_news_md / pl_core_news_lg
# components : tok2vec, morphologizer, parser, tagger, senter, ner, attribute_ruler, lemmatizer
nlp = spacy.load('pl_core_news_sm') 
nlp.pipeline
nlp.pipe_names

# doc = nlp("Prognozy wskazują, że w najbliższych dniach może zostać pobity w Polsce rekord zimna. Temperatura ma spaść nawet do –40 stopni Celsjusza. Ostatni raz podobne mrozy odnotowano w naszym kraju w 1929 roku.")
doc = nlp("Naukowcy przyglądają się lodowcowi szelfowemu Brunt na Antarktydzie od początku 2019 r. Eksperci spodziewają się, że oderwanie gigantycznej góry lodowej nastąpi w najbliższym czasie, po tym jak pojawiły się nowe pęknięcia.")
displacy.serve(doc, style="dep") # http://localhost:5000
displacy.serve(doc, style="ent")
# POS, lemmas http://stanza.run
# POS tagger API http://clarin.pelcra.pl/tools/tagger/ | https://krnnt-f3esrhez2q-ew.a.run.app/
pddoc = display_nlp(doc)
# print([(w.text, w.pos_) for w in doc])

nouns = [t.lemma_ for t in doc if t.pos_ in ['NOUN']]

patterns = ["POS:ADJ POS:NOUN:+"]
spans = textacy.extract.matches(doc, patterns=patterns)
print(*[s.lemma_ for s in spans], sep='|')

for col, values in extract_nlp(doc).items():
 print(f"{col}: {values}")

# doc = nlp("One czytała książkę") #morphology
print(doc[0].morph)
print(doc[0].pos_)
print(doc[0].morph.get("PronType"))

print([token.lemma_ for token in doc]) #Lemmatization

# Named Entity Recognition sprawdzic https://github.com/CLARIN-PL/PolDeepNer
# Training NER model https://github.com/practical-nlp/practical-nlp/blob/master/Ch5/04_NER_using_spaCy%20-%20CoNLL.ipynb
# To illustrate the difference between a normal classifier and asequence classifier, consider the following sentence: “Washington is arainy state.” When a normal classifier sees this sentence and has toclassify it word by word, it has to make a decision as to whetherWashington refers to a person (e.g., George Washington) or the Stateof Washington without looking at the surrounding words. It’s possibleto classify the word “Washington” in this particular sentence as alocation only after looking at the context in which it’s being used. It’sfor this reason that sequence classifiers are used for training NER models
for ent in doc.ents: # NER
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
for sent in doc.sents: #sentence segmenter
    print(sent.text)
    
# Word vectors and semantic similarity
# It needs lg model for word vector

# ADJ   Adjectives (describe nouns) big, green, African
# ADP   Adpositions (prepositions and postpositions) in, on
# ADV   Adverbs (modify verbs or adjectives) very, exactly, always
# AUX   Auxiliary (accompanies verb) can (do), is (doing)
# CCONJ Connecting conjunction and, or, but
# DET   Determiner (with regard to nouns) the, a, all (things), your (idea)
# INTJ  Interjection (independent word, exclamation, expression of emotion) hi, yeah
# NOUN  Nouns (common and proper) house, computer
# NUM   Cardinal numbers nine, 9, IX
# PROPN Proper noun, name, or part of a name Peter, Berlin
# PRON  Pronoun, substitute for noun I, you, myself, who
# PART  Particle (makes sense only with other word)
# PUNCT Punctuation characters , . ;
# SCONJ Subordinating conjunction before, since, if
# SYM   Symbols (word-like) $, ©
# VERB  Verbs (all tenses and modes) go, went, thinking
# X     Anything that cannot be assigned grlmpf


# NLTK part-of-speech tagger
import nltk
from nltk.corpus.reader import pl196x
nltk.download()
pl196x_dir = nltk.data.find('C:/Users/Kamil/AppData/Roaming/nltk_data/corpora/pl196x')
pl = pl196x.Pl196xCorpusReader(pl196x_dir,r'.*\.xml',textids='textids.txt',cat_file="cats.txt")
print(pl.fileids())
twords = pl.tagged_words(fileids=pl.fileids(),categories='cats.txt')
for w in twords[:10]:
 print(w)
 
tsents = pl.tagged_sents(fileids=pl.fileids(),categories='cats.txt')[:3000]
tagger = nltk.UnigramTagger(tsents)

tekst = "Prognozy wskazują, że w najbliższych dniach może zostać pobity w Polsce rekord zimna. Temperatura ma spaść nawet do –40 stopni Celsjusza. Ostatni raz podobne mrozy odnotowano w naszym kraju w 1929 roku."
tagger.tag(tekst.split())

test_sents = pl.tagged_sents(fileids=pl.fileids(),categories='cats.txt')[3000:6000]
tagger.evaluate(test_sents)


# nltk lemmatization PlWordNet nie działa z nową wersja
# mozna wykorzystac Morfeusz python API 

# from nltk.corpus import wordnet as wn
# wn.synsets('Politechnika')
# wn.langs()


# Word Sense Disambiguation lepiej użyć model state-of-the-art np BERT
from nltk.wsd import lesk 
from nltk import word_tokenize 
samples = [('The fruits on that plant have ripened', 'n'),
 ('He finally reaped the fruit of his hard work as he won the race', 'n')]
word = 'fruit'

for sentence, pos_tag in samples:
 word_syn = lesk(word_tokenize(sentence.lower()), word, pos_tag)
 print ('Sentence:', sentence)
 print ('Word synset:', word_syn)
 print ('Corresponding definition:', word_syn.definition() )
 
 
