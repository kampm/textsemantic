# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:10:13 2021

@author: Kamil
"""

# sentence models comparation https://github.com/sdadas/polish-sentence-evaluation

# model
# pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.1/xx_distiluse_base_multilingual_cased-0.1.1.tar.gz#xx_distiluse_base_multilingual_cased-0.1.1
import spacy
import spacy_sentence_bert

nlp = spacy_sentence_bert.load_model("xx_distiluse_base_multilingual_cased")

s1 = nlp('Piłka nożna z wieloma grającymi facetami')
s2 = nlp('Jacyś mężczyźni grają w futbol')
s3 = nlp('Kobiety idą do fryzjera')
senTab = [s1, s2, s3]

# get the vector of the Doc, Span or Token
print(s1.vector.shape)
print(s1[3].vector.shape)
print(s1[2:4].vector.shape)
# or use the similarity method that is based on the vectors, on Doc, Span or Token

for i in senTab:
    print(f'{i} | {s1} = {i.similarity(s1)}')
    print(f'{i} | {s2} = {i.similarity(s2)}')
    print(f'{i} | {s3} = {i.similarity(s3)}')

# Piłka nożna z wieloma grającymi facetami | Piłka nożna z wieloma grającymi facetami = 1.0
# Piłka nożna z wieloma grającymi facetami | Jacyś mężczyźni grają w futbol = 0.7743951213403814
# Piłka nożna z wieloma grającymi facetami | Kobiety idą do fryzjera = 0.18865164464797418
# Jacyś mężczyźni grają w futbol | Piłka nożna z wieloma grającymi facetami = 0.7743951213403814
# Jacyś mężczyźni grają w futbol | Jacyś mężczyźni grają w futbol = 1.0
# Jacyś mężczyźni grają w futbol | Kobiety idą do fryzjera = 0.22059777290427954
# Kobiety idą do fryzjera | Piłka nożna z wieloma grającymi facetami = 0.18865164464797418
# Kobiety idą do fryzjera | Jacyś mężczyźni grają w futbol = 0.22059777290427954
# Kobiety idą do fryzjera | Kobiety idą do fryzjera = 1.0