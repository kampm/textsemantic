-*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:06:45 2021

@author: Kamil
"""
# https://arxiv.org/pdf/2006.04229.pdf
# QA
from transformers import *
# QA model https://huggingface.co/henryk/bert-base-multilingual-cased-finetuned-polish-squad2
qa_pipeline = pipeline('question-answering',
    model="bert-base-multilingual-cased-finetuned-polish-squad2",
    tokenizer="bert-base-multilingual-cased-finetuned-polish-squad2"
)
qa_pipeline({
    'context': "Warszawa jest największym miastem w Polsce pod względem liczby ludności i powierzchni",
    'question': "Jakie jest największe miasto w Polsce?"})

#MaskedLM https://huggingface.co/dkleczek/bert-base-polish-cased-v1
model = BertForMaskedLM.from_pretrained(
    "dkleczek/bert-base-polish-cased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
# model = AutoModelWithLMHead.from_pretrained("roberta_base")
# tokenizer = AutoTokenizer.from_pretrained("roberta_base")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)

# sequence = "The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas early enough in the evening to have a nice dinner and go see a show."
# nlp_qa(context=sequence, question='Where is Pioneer Boulevard ?')

# BERT NER Example training model from pretrained https://github.com/practical-nlp/practical-nlp/blob/master/Ch5/05_BERT_CONLL_NER.ipynb
# NER https://demo.allennlp.org/named-entity-recognition/fine-grained-ner
nlp_ner = pipeline("fill-mask",model="bert-base-multilingual-cased-finetuned-polish-squad2",
    tokenizer="bert-base-multilingual-cased-finetuned-polish-squad2"
)
sequence="Ruch na Pioneer Boulevard w Los Angeles zaczął zwalniać, co utrudniało wydostanie się z miasta. Jednak WBGO grało fajny jazz, a pogoda była chłodna, co sprawiało, że wyjechanie z miasta w to piątkowe popołudnie było raczej przyjemne. Nat King Cole śpiewał, gdy Jo i Maria powoli wyjeżdżały z Los Angeles i jechały w kierunku Barstow. Planowali dotrzeć do Las Vegas na tyle wcześnie wieczorem, aby zjeść miłą kolację i obejrzeć przedstawienie."
qa_pipeline(context=sequence, question='Gdzie jest LA ?')
qa_pipeline(context=sequence, question='Gdzie jest Pioneer Boulevard ?')
qa_pipeline(context=sequence, question='Gdzie znajduje się Pioneer Boulevard ?')
qa_pipeline(context=sequence, question='Gdzie położony jest Pioneer Boulevard ?')
qa_pipeline(context=sequence, question='Jaka jest lokalizacja Pioneer Boulevard ?')
qa_pipeline(context=sequence, question='Gdzie znajduje się Las Vegas ?')
qa_pipeline(context=sequence, question='Kto śpiewał?') # 'score': 0.9957380890846252
qa_pipeline(context=sequence, question='Kto spiewał?') # 'score': 0.20145834982395172
qa_pipeline(context=sequence, question='Kto przyjechał do Las Vegas?')
print(nlp_ner(sequence))

for pred in nlp_ner(f"Adam Mickiewicz wielkim polskim {nlp_ner.tokenizer.mask_token} był."):
  print(pred)


3CosAdd “what is to Portugal as Paris is to France?”. 
b∗ = argmax cos(w, a∗ − a + b) 
w∈V

3CosMul
b∗ = argmax ( cos(b, w) × cos(w, a∗) / cos(w, a) )
w∈V


# not working!
# MAX_LENGTH = 120 #@param {type: "integer"}
# MODEL = "roberta_base"

# import sys

# from transformers import AutoTokenizer

# dataset = "train_temp.txt"
# model_name_or_path = MODEL
# max_len = 120

# subword_len_counter = 0

# import torch, os
# from transformers import RobertaModel, AutoModel, PreTrainedTokenizerFast
# tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))


# def load_data(filename: str):
#     with open(filename, 'r', encoding="utf-8") as file:
#         lines = [line[:-1].split() for line in file if "-DOCSTART" not in line]
#     samples, start = [], 0
#     for end, parts in enumerate(lines):
#         if not parts:
#             try:
#                 sample = [(token, tag)
#                           for token, tag in lines[start:end]]
#             except:
#                 print(lines[start:end])
#             samples.append(sample)
#             start = end + 1
#     if start < end:
#         samples.append(lines[start:end])
#     return samples


# train_samples = load_data(dataset)
# val_samples = train_samples[:8000]# test

# samples = train_samples 
# schema = ['_'] + sorted({tag for sentence in samples
#                          for _, tag in sentence})

# import tqdm
# import numpy as np
# def tokenize_sample(sample):
#     seq = [
#         (subtoken, tag)
#         for token, tag in sample
#         for subtoken in tokenizer(token)['input_ids'][1:-1]
#     ]
#     return [(3, 'O')] + seq + [(4, 'O')]


# def preprocess(samples):
#     tag_index = {tag: i for i, tag in enumerate(schema)}
#     tokenized_samples = list(map(tokenize_sample, samples))
#     max_len = max(map(len, tokenized_samples))
#     X = np.zeros((len(samples), max_len), dtype=np.int32)
#     y = np.zeros((len(samples), max_len), dtype=np.int32)
#     for i, sentence in enumerate(tokenized_samples):
#         for j, (subtoken_id, tag) in enumerate(sentence):
#             X[i, j] = subtoken_id
#             y[i, j] = tag_index[tag]
#     return X, y


# X_train, y_train = preprocess(train_samples)
# X_val, y_val = preprocess(val_samples)

# print(X_train.shape)
# print(y_train.shape)
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))

# import pandas as pd

# EPOCHS=10
# BATCH_SIZE=16

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense


# def build_model(nr_filters=256):
#     input_shape = (1, 141)
#     # lstm = LSTM(nr_filters, return_sequences=True)
#     lstm =LSTM(units=50, input_shape=(1, 141), return_sequences=True)
#     bi_lstm = Bidirectional(lstm, input_shape=input_shape)
#     tag_classifier = Dense(len(schema), activation='softmax')
#     sequence_labeller = TimeDistributed(tag_classifier)
#     return Sequential([bi_lstm, sequence_labeller])

# model = build_model()
# model.summary()



# optimizer = tf.keras.optimizers.Adam(lr=0.00001)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# history = model.fit(tf.constant(X_train), tf.constant(y_train),
#                     validation_split=0.2, epochs=EPOCHS, 
#                     batch_size=BATCH_SIZE)




##NERDA it should work but it gives OOM (4gb mem)
#it works here -> https://www.kaggle.com/kkkkkkk880/ner-bert-base-polish-cased-v1/
# !!!!!!!!!!!!!!!!!!

count=0
countdoc=0
tabtrain=[]
tabtest=[]
tabvalid=[]
with open("nkjp_ner/nkjp-nested-simplified-v2.txt", encoding="utf-8") as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        if "-DOCSTART" in line:
            countdoc += 1
            line = ""
        if countdoc<3200:
            tabtrain.append(line.replace("	"," ").replace("#"," "))
        
        elif 3000<countdoc<3700:
            tabtest.append(line.replace("	"," ").replace("#"," "))
        elif countdoc>3700:
            tabvalid.append(line.replace("	"," ").replace("#"," "))
    
fa = open(r"nkjp_ner/train.txt","w+", encoding="utf-8")
fa.write("".join(tabtrain))
fa.close()

fa = open(r"nkjp_ner/valid.txt","w+", encoding="utf-8")
fa.write("".join(tabvalid))
fa.close()

fa = open(r"nkjp_ner/test.txt","w+", encoding="utf-8")
fa.write("".join(tabtest))
fa.close()


from NERDA.datasets import get_dane_data,get_conll_data, download_conll_data 




# download_conll_data()
training = get_conll_data('train')
validation = get_conll_data('valid')
test = get_conll_data('test')
training["sentences"][0]



unique_data = list(map(list, set(map(lambda i: tuple(i), training["tags"]))))
result = list(set(x for l in unique_data for x in l))
for r in result:
    print(f'"{r}",')
    
unique_data = list(map(list, set(map(lambda i: tuple(i), test["tags"]))))
result2 = list(set(x for l in unique_data for x in l))

tag_scheme = [
"B-persName-forename",
"I-placeName-bloc",
"B-orgName",
"I-placeName-region",
"B-persName-surname",
"B-placeName",
"B-date",
"B-time",
"I-persName",
"B-geogName",
"I-placeName-settlement",
"I-date",
"I-persName-surname",
"B-persName",
"B-placeName-region",
"I-geogName",
"I-orgName",
"B-placeName-country",
"I-persName-forename",
"I-time",
"I-placeName-country",
"B-placeName-bloc",
"I-persName-addName",
"B-placeName-district",
"B-placeName-settlement",
"I-placeName-district",
"B-persName-addName"
]

transformer = 'dkleczek/bert-base-polish-cased-v1'# only bert small models/roberta OOM



# hyperparameters for network
dropout = 0.1
# hyperparameters for training
training_hyperparameters = {
'epochs' : 1,'warmup_steps' : 500,'train_batch_size': 1,'learning_rate': 0.0001
}

from NERDA.models import NERDA
model = NERDA(
dataset_training = training,
dataset_validation = validation,
tag_scheme = tag_scheme, 
tag_outside = 'O',
transformer = transformer,
dropout = dropout,
hyperparameters = training_hyperparameters
)

model.train()

# test = get_conll_data('test')
# model.evaluate_performance(test)

#eng train data on pl bert
# Level	F1-Score
# 0	B-PER	0.884507
# 1	I-PER	0.941590
# 2	B-ORG	0.771963
# 3	I-ORG	0.751755
# 4	B-LOC	0.864624
# 5	I-LOC	0.704762
# 6	B-MISC	0.738714
# 7	I-MISC	0.570136
# 0	AVG_MICRO	0.825114
# 0	AVG_MACRO	0.778506

#pl train data on pl bert (eval-vaild.txt)
e1 Train Loss = 0.26577291781040807 Valid Loss = 0.12885032495519444
e2 Train Loss = 0.08480613472856427 Valid Loss = 0.10683761777700092
e3 Train Loss = 0.04463438292757332 Valid Loss = 0.11588288823524333
e4 Train Loss = 0.024082582885794748 Valid Loss = 0.11854389137669845
e5 Train Loss = 0.011541870708788758 Valid Loss = 0.11877383219411418
# 	Level	F1-Score
# 0	B-persName	0.673611
# 1	B-placeName-bloc	0.125000
# 2	B-persName-forename	0.917184
# 3	I-date	0.953954
# 4	B-placeName-region	0.686391
# 5	I-persName-forename	0.797688
# 6	B-placeName	0.861702
# 7	B-geogName	0.631502
# 8	B-persName-surname	0.915133
# 9	I-placeName-settlement	0.733813
# 10	I-placeName-bloc	0.000000
# 11	I-placeName	0.000000
# 12	B-date	0.925566
# 13	I-placeName-region	0.861925
# 14	I-persName-addName	0.642857
# 15	B-placeName-country	0.901505
# 16	I-placeName-district	0.230769
# 17	I-persName-surname	0.905660
# 18	B-persName-addName	0.262774
# 19	I-time	0.925490
# 20	I-orgName	0.772438
# 21	B-placeName-settlement	0.872242
# 22	I-geogName	0.682051
# 23	B-time	0.852459
# 24	I-persName	0.801724
# 25	I-placeName-country	0.691589
# 26	B-placeName-district	0.383838
# 27	B-orgName	0.739605
# 0	AVG_MICRO	0.836549
# 0	AVG_MACRO	0.669588

#pl train data on pl bert (eval-test.txt)
# Level	F1-Score
# 0	B-persName	0.673913
# 1	B-placeName-bloc	0.333333
# 2	B-persName-forename	0.873257
# 3	I-date	0.854881
# 4	B-placeName-region	0.500000
# 5	I-persName-forename	0.142857
# 6	B-placeName	0.769231
# 7	B-geogName	0.457143
# 8	B-persName-surname	0.895307
# 9	I-placeName-settlement	0.571429
# 10	I-placeName-bloc	0.000000
# 11	I-placeName	0.000000
# 12	B-date	0.753846
# 13	I-placeName-region	1.000000
# 14	I-persName-addName	0.000000
# 15	B-placeName-country	0.916808
# 16	I-placeName-district	0.000000
# 17	I-persName-surname	0.000000
# 18	B-persName-addName	0.202532
# 19	I-time	0.847059
# 20	I-orgName	0.809524
# 21	B-placeName-settlement	0.750000
# 22	I-geogName	0.473684
# 23	B-time	0.813953
# 24	I-persName	0.564103
# 25	I-placeName-country	0.666667
# 26	B-placeName-district	0.500000
# 27	B-orgName	0.700201
# 0	AVG_MICRO	0.795547
# 0	AVG_MACRO	0.538205