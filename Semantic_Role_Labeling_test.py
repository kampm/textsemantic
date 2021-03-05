# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:04:13 2021

@author: Kamil
"""
# better to use spacy POS tagging

# https://demo.allennlp.org/semantic-role-labeling
# NLP research library, built on PyTorch and spaCy
from allennlp_models.structured_prediction.models import srl_bert
from allennlp.modules.token_embedders import BertEmbedder
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# ✔ Download and installation successful
# You can now load the model via spacy.load('en_core_web_sm')

text = "Did Bob really think he could prepare a meal for 50 people in only a few hours?"
predictor.predict(sentence=text)

