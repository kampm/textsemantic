# -*- coding: utf-8 -*-
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


# 3CosAdd “what is to Portugal as Paris is to France?”. 
# b∗ = argmax cos(w, a∗ − a + b) 
# w∈V

# 3CosMul
# b∗ = argmax ( cos(b, w) × cos(w, a∗) / cos(w, a) )
# w∈V

