# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:20:18 2021

@author: Kamil
"""

import torch
from transformers import RobertaForMaskedLM,LineByLineTextDataset,RobertaTokenizer, BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelWithLMHead, ElectraModel,PreTrainedModel ,ElectraForMaskedLM
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

#train=hate_speech_pl,block_size=100,epochs=5,batch=4
#chodzi mi o normalnych <mask> a nie ortodoksów = _katolików,▁chrześcijan,_ludzi,_obywateli,▁prawosławnych
#awanturujący się <mask> to przestępcy = ▁ludzie,eje,_,_gej,_policjanci
# <mask> są bezwstydne i nie mają granic i hamluców = _kobiety,eje,_one,_gej,_baby

#train=CDS_corpus,block_size=256,epochs=5,batch=4
# <mask> w czerwonej bluzie stoi przy barierce nad wodą , a obok niego stoi oparta wędka . = Człowiek,Para,▁mężczyzna,Ludzie,▁Mężczyzna
#Sześć osób <mask> w hokeja na lodzie . = _gra,_uprawia,_grają,skakuje,łuje