# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:20:18 2021

@author: Kamil
"""

# RoBERTa has exactly the same architecture as BERT. The only differences are:

# RoBERTa uses a Byte-Level BPE tokenizer with a larger subword vocabulary (50k vs 32k).
# RoBERTa implements dynamic word masking and drops next sentence prediction task.
# RoBERTa's training hyperparameters.

import torch
from transformers import RobertaForMaskedLM,LineByLineTextDataset,RobertaTokenizer, BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelWithLMHead,AutoModelForMaskedLM, ElectraModel,PreTrainedModel ,ElectraForMaskedLM
from transformers import Trainer,logging, TrainingArguments,pipeline,LineByLineTextDataset,DataCollatorForLanguageModeling
MODEL_PATH = 'roberta_base'


# VOCAB = MODEL_PATH

# print('== tokenizing ===')
# # tokenizer = BertTokenizer.from_pretrained(VOCAB)
# tokenizer = AutoTokenizer.from_pretrained(VOCAB)
# # Tokenized input
# text = "Kim był Jim Henson ? Jim [MASK] był kierowcą"
# inputs = tokenizer.encode_plus(text, return_tensors="pt")

# masked_index = 7

# model = BertForMaskedLM.from_pretrained(MODEL_PATH)
# model.eval()

# print('== LM predicting ===')
# # Predict all tokens
# predictions = model(**inputs)[0]

# # confirm we were able to predict 'henson'
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
# print('predicted_token', predicted_token)



# fill_mask = pipeline(
#     "fill-mask",
#     model=MODEL_PATH,
#     tokenizer=MODEL_PATH
# )
# fill_mask("Wylij te <mask> spowrotem!")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelWithLMHead.from_pretrained(MODEL_PATH)



# from datasets import load_dataset,list_datasets
# from bs4 import BeautifulSoup 
# dataset = load_dataset('hate_speech_pl',split='train')

# datasets_list = list_datasets()
# data= dataset['text']
# data2=[]
# for s in data:
#     data2.append(BeautifulSoup(s).get_text().replace("gt ","").replace("  "," ").replace("\n", " ").lower()+"\n")# or skip to lowercase?
# data2="".join(data2)
# f = open(r"train_text_pl.txt","w+", encoding="utf-8")
# f.write(data2)
# f.close()

# f = open(r"roberta_base/merges.txt","r", encoding="utf-8")
# Lines = f.readlines()
# ss=[x.replace('"','') for xs in Lines for x in xs.split('","')]
# tem=[]
# for s in ss:
#     tem.append(s+"\n")# or skip to lowercase?


# fa = open(r"roberta_base/merges2.txt","w+", encoding="utf-8")
# fa.write("".join(tem))
# fa.close()

# import pandas as pd
# corpus = pd.read_csv('CDSCorpus/CDS_train.csv',sep='\t',error_bad_lines=False, encoding='utf-8')# ,nrows=1000  
# s=pd.DataFrame(pd.unique(corpus[['sentence_A', 'sentence_B']].values.ravel('K')))
# s.to_csv('CDS_uniq_mlm.txt',sep="\n",header=False,index=False)


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    # file_path='train_text_pl.txt',
    file_path='CDS_uniq_mlm.txt',
    block_size=256,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./roberta_base_CDS_mlm",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    seed=1
)
torch.cuda.empty_cache()
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

logging.set_verbosity_info()
trainer.train()


trainer.save_model("./roberta_base_CDS_mlm")

fill_mask = pipeline(
    "fill-mask",
    model="./roberta_base_CDS_mlm",
    tokenizer=MODEL_PATH
)
fill_mask("Sześć osób <mask> w hokeja na lodzie .")

# for pred in fill_mask(f"Sześć osób {fill_mask.tokenizer.mask_token} w hokeja na lodzie ."):
#   print(pred)

# https://www.kaggle.com/kkkkkkk880/masklm-bert-hatepl
#train=hate_speech_pl,block_size=100,epochs=5,batch=4
#chodzi mi o normalnych <mask> a nie ortodoksów = _katolików,▁chrześcijan,_ludzi,_obywateli,▁prawosławnych
#awanturujący się <mask> to przestępcy = ▁ludzie,eje,_,_gej,_policjanci
# <mask> są bezwstydne i nie mają granic i hamluców = _kobiety,eje,_one,_gej,_baby

#train=hate_speech_pl,block_size=500,epochs=5,batch=8
#chodzi mi o normalnych <mask> a nie ortodoksów = ▁ludzi,▁chrześcijan,▁katolików,▁obywateli,▁facetów
#awanturujący się <mask> to przestępcy = ▁ludzie,_gej,_arab,_mężczyżni,_politycy
# <mask> są bezwstydne i nie mają granic i hamluców = _kobiety,_one,_,_która,_dzieci
#<mask> są bezwstydne i nie mają granic i hamluców = które,oni,_one,słowa,ty


#train=CDS_corpus,block_size=256,epochs=5,batch=4
# <mask> w czerwonej bluzie stoi przy barierce nad wodą , a obok niego stoi oparta wędka . = Człowiek,Para,▁mężczyzna,Ludzie,▁Mężczyzna
#Sześć osób <mask> w hokeja na lodzie . = _gra,_uprawia,_grają,skakuje,łuje


##################################################wiki
from tokenizers import ByteLevelBPETokenizer

path = "roberta_test/train.txt" #plwiki

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=path,
                vocab_size=50265,
                min_frequency=5,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])



tokenizer.save("roberta_test/tokenizer.json")

import json
config = {
	"architectures": [
		"RobertaForMaskedLM"
	],
	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"layer_norm_eps": 1e-05,
	"max_position_embeddings": 514,
	"model_type": "roberta",
	"num_attention_heads": 12,
	"num_hidden_layers": 12,
	"type_vocab_size": 1,
	"vocab_size": 50265
}

with open("roberta_test/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {"max_len": 126}

with open("roberta_test/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)
    
MODEL_TYPE = "roberta" #@param ["roberta", "bert"]
MODEL_DIR = "roberta_test" #@param {type: "string"}
OUTPUT_DIR = "roberta_test/output" #@param {type: "string"}
TRAIN_PATH = "roberta_test/train.txt" #@param {type: "string"}
EVAL_PATH = "roberta_test/dev.txt" #@param {type: "string"}


# cmd = """python run_language_modeling.py \
#     --output_dir {output_dir} \
#     --model_type {model_type} \
#     --mlm \
#     --config_name {config_name} \
#     --tokenizer_name {tokenizer_name} \
#     {line_by_line} \
#     {should_continue} \
#     {model_name_or_path} \
#     --train_data_file {train_path} \
#     --eval_data_file {eval_path} \
#     --do_train \
#     {do_eval} \
#     {evaluate_during_training} \
#     --overwrite_output_dir \
#     --block_size 126 \
#     --max_step 25 \
#     --warmup_steps 10 \
#     --learning_rate 5e-5 \
#     --per_gpu_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --weight_decay 0.01 \
#     --adam_epsilon 1e-6 \
#     --max_grad_norm 100.0 \
#     --save_total_limit 10 \
#     --save_steps 10 \
#     --logging_steps 2 \
#     --seed 42
# """

cmd = """python run_mlm.py \
    --output_dir {output_dir} \
    --model_type {model_type} \
    --config_name {config_name} \
    --tokenizer_name {tokenizer_name} \
    {line_by_line} \
    {should_continue} \
    {model_name_or_path} \
    --train_file {train_path} \
    --validation_file {eval_path} \
    --do_train \
    {do_eval} \
    {evaluate_during_training} \
    --overwrite_output_dir \
    --max_seq_length 126 \
    --max_step 25 \
    --warmup_steps 10 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --save_total_limit 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --seed 42
"""

train_params = {
    "output_dir": OUTPUT_DIR,
    "model_type": MODEL_TYPE,
    "config_name": MODEL_DIR,
    "tokenizer_name": MODEL_DIR,
    "train_path": TRAIN_PATH,
    "eval_path": EVAL_PATH,
    "do_eval": "--do_eval",
    "evaluate_during_training": "",
    "line_by_line": "",
    "should_continue": "",
    "model_name_or_path": "",
}

!{cmd.format(**train_params)}


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="roberta_test/output",
    tokenizer="roberta_test/output"
)

fill_mask("interpretowany język <mask>.")

#########################################Extract-text-from-WolneLekturyPL.ipynb

import json
from pathlib import Path
from glob import glob
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import nltk
import re
nltk.download('punkt')

extra_abbreviations = ['ps',  'inc', 'Corp', 'Ltd', 'Co', 'pkt', 'Dz.Ap', 'Jr', 'jr', 'sp', 'Sp', 'poj',  'pseud', 'krypt', 'sygn', 'Dz.U', 'ws', 'itd', 'np', 'sanskryt', 'nr', 'gł', 'Takht', 'tzw', 't.zw', 'ewan', 'tyt', 'oryg', 't.j', 'vs', 'l.mn', 'l.poj' ]

position_abbrev = ['Ks', 'Abp', 'abp','bp','dr', 'kard', 'mgr', 'prof', 'zwycz', 'hab', 'arch', 'arch.kraj', 'B.Sc', 'Ph.D', 'lek', 'med', 'n.med', 'bł', 'św', 'hr', 'dziek' ]

quantity_abbrev = [ 'mln', 'obr./min','km/godz', 'godz', 'egz', 'ha', 'j.m', 'cal', 'obj', 'alk', 'wag' ] # not added: tys.

actions_abbrev = ['tłum','tlum','zob','wym', 'pot', 'ww', 'ogł', 'wyd', 'min', 'm.i', 'm.in', 'in', 'im','muz','tj', 'dot', 'wsp', 'właść', 'właśc', 'przedr', 'czyt', 'proj', 'dosł', 'hist', 'daw', 'zwł', 'zaw' ]

place_abbrev = ['Śl', 'płd', 'geogr']

lang_abbrev = ['jęz','fr','franc', 'ukr', 'ang', 'gr', 'hebr', 'czes', 'pol', 'niem', 'arab', 'egip', 'hiszp', 'jap', 'chin', 'kor', 'tyb', 'wiet', 'sum']

military_abbrev = ['kpt', 'kpr', 'obs', 'pil', 'mjr','płk', 'dypl', 'pp', 'gw', 'dyw', 'bryg', 'ppłk', 'mar', 'marsz', 'rez', 'ppor', 'DPanc', 'BPanc', 'DKaw', 'p.uł']

extra_abbreviations= extra_abbreviations + position_abbrev + quantity_abbrev + place_abbrev + actions_abbrev + place_abbrev + lang_abbrev+military_abbrev

sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
sent_tokenize = sentence_tokenizer.tokenize

def flatten(iterable):
    return chain.from_iterable(iterable)

def preprocess_book(book_txt):
    if "404 Not Found" in book_txt:
        return ""
    
    start_idx = book_txt.index("\n" * 4) + 5
    end_idx = book_txt.index("-----") - 5
    txt =  book_txt[start_idx: end_idx]
    return re.sub("\s+", " ", txt)

def process_book(book_path):
    
    try:
        txt = preprocess_book(Path(book_path).read_text("utf-8"))
        sentences = [s for s in sent_tokenize(txt) if len(s) >= 16]
        windowed_sentences = []
        for snt in range(len(sentences)):
            windowed_sentences.append(" ".join(sentences[snt: snt + 8]))
        return windowed_sentences
    except:
        print(f"Could not parse \n{book_path}\n")
        return []

books = list(glob("wolne-lektury/*.txt"))
books[:10]

buffer, BUFFER_SIZE = [], 100000
with open("wolne-lektury.sliding8.txt", "wt",encoding="utf-8") as file:
    for i, sentence in enumerate(flatten(process_book(f) for f in books)):
        if len(buffer) >= BUFFER_SIZE:
            file.write("\n".join(buffer))
            buffer.clear()
            print(i, end="\r")
        buffer.append(sentence)
    if len(buffer) > 0:
        file.write("\n".join(buffer))
        buffer.clear()
     
        
     
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from glob import glob
paths = list(
    glob("wolne-lektury.sliding8.txt")
)
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=False)

# Customize training
tokenizer.train(files=paths, vocab_size=32000, min_frequency=3, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])



tokenizer.save("wolne-lektury/tokenizer.json")

import json
config = {
	"architectures": [
		"RobertaForMaskedLM"
	],
	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"layer_norm_eps": 1e-05,
	"max_position_embeddings": 514,
	"model_type": "roberta",
	"num_attention_heads": 12,
	"num_hidden_layers": 12,
	"type_vocab_size": 1,
	"vocab_size": 32000
}

with open("wolne-lektury/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {"max_len": 126}

with open("wolne-lektury/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)

# Save files to disk
# import os
# OUT_DIR = "polish_tokenizer_bpe_32k"
# os.makedirs(OUT_DIR, exist_ok=True)
# tokenizer.save(OUT_DIR, "pl")
MODEL_TYPE = "roberta" #@param ["roberta", "bert"]
MODEL_DIR = "wolne-lektury" #@param {type: "string"}
OUTPUT_DIR = "wolne-lektury/output" #@param {type: "string"}
TRAIN_PATH = "wolne-lektury/wolne-lektury.sliding8.txt" #@param {type: "string"}
EVAL_PATH = "wolne-lektury/dev.txt" #@param {type: "string"}

cmd = """python run_mlm.py \
    --output_dir {output_dir} \
    --model_type {model_type} \
    --config_name {config_name} \
    --tokenizer_name {tokenizer_name} \
    {line_by_line} \
    {should_continue} \
    {model_name_or_path} \
    --train_file {train_path} \
    --validation_file {eval_path} \
    --do_train \
    {do_eval} \
    {evaluate_during_training} \
    --overwrite_output_dir \
    --max_seq_length 126 \
    --max_step 25 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --save_total_limit 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --seed 42
"""

train_params = {
    "output_dir": OUTPUT_DIR,
    "model_type": MODEL_TYPE,
    "config_name": MODEL_DIR,
    "tokenizer_name": MODEL_DIR,
    "train_path": TRAIN_PATH,
    "eval_path": EVAL_PATH,
    "do_eval": "--do_eval",
    "evaluate_during_training": "",
    "line_by_line": "",
    "should_continue": "",
    "model_name_or_path": "",
}

!{cmd.format(**train_params)}


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="roberta_test/output",
    tokenizer="roberta_test/output"
)

# fill_mask("Poszedł i nawet zdał <mask> do Szkoły Głównej")
# predicted words: siatk,*,wskaźników,kolory,lizmu


#https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
#the result for the argument max_seq_length-126 gives very weak predictions
# allplwiki requires 500gb ram/ for more sequences(max_seq_length)  you need much more gpu memory at least 8gb