# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:16:37 2021

@author: Kamil
"""
# pl model https://github.com/sdadas/polish-nlp-resources
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer # Hugging Face’s

paths = [str(x) for x in Path(".").glob("**/*.txt")]
paths =["kant.txt"]
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
# <s> : a start token
# <pad> : a padding token
# </s> : an end token
# <unk> : an unknown token
# <mask> : the mask token for language modeling


import os
token_dir = '/content/KantaiBERT'
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model('F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT')


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT/vocab.json",
    "F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT/merges.txt",
)

tokenizer.encode("The Critique of Pure Reason.").tokens

tokenizer.encode("The Critique of Pure Reason.")

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

import torch
torch.cuda.is_available()

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT", max_length=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
# print(model)
LP=list(model.parameters())
lp=len(LP)
print(lp)
for p in range(0,lp):
  print(LP[p])

np=0
for p in range(0,lp):#number of tensors
  PL2=True
  try:
    L2=len(LP[p][0]) #check if 2D
  except:
    L2=1             #not 2D but 1D
    PL2=False
  L1=len(LP[p])      
  L3=L1*L2
  np+=L3             # number of parameters per tensor
  if PL2==True:
    print(p,L1,L2,L3)  # displaying the sizes of the parameters
  if PL2==False:
    print(p,L1,L3)  # displaying the sizes of the parameters

print(np)   

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="kant.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# TrainOutput(global_step=10686, training_loss=3.813465910191708, metrics={'train_runtime': 1332.6534, 'train_samples_per_second': 8.019, 'total_flos': 1571537744307456, 'epoch': 1.0})

trainer.save_model("F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT",
    tokenizer="F:/PycharmProjects/zajecia/spyder/semantic/textsemantic/content/KantaiBERT"
)

fill_mask("Human thinking involves<mask>.")
# [{'sequence': 'Human thinking involves it.',
#   'score': 0.10453205555677414,
#   'token': 306,
#   'token_str': ' it'},
#  {'sequence': 'Human thinking involves them.',
#   'score': 0.019846992567181587,
#   'token': 508,
#   'token_str': ' them'},
#  {'sequence': 'Human thinking involves experience.',
#   'score': 0.015704313293099403,
#   'token': 531,
#   'token_str': ' experience'},
#  {'sequence': 'Human thinking involves us.',
#   'score': 0.013465079478919506,
#   'token': 538,
#   'token_str': ' us'},
#  {'sequence': 'Human thinking involves itself.',
#   'score': 0.011306803673505783,
#   'token': 500,
#   'token_str': ' itself'}]