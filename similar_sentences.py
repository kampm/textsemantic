# https://arxiv.org/pdf/1911.02929.pdf
# https://www.aclweb.org/anthology/2020.coling-main.300.pdf

# from transformers import BertModel, BertTokenizer

# model_class = BertModel
# tokenizer_class = BertTokenizer
# pretrained_weights = 'bert-base-multilingual-cased-finetuned-polish-squad2'

# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)

# sentence = 'to jest tekst'
# input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
# with torch.no_grad():
#     output_tuple = model(input_ids)
#     last_hidden_states = output_tuple[0]
# print(last_hidden_states.size(), last_hidden_states)


# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  
# from transformers import pipeline

# fill_mask = pipeline(
#     "fill-mask",
#     model="roberta_large",
#     tokenizer="roberta_large"
# )

# kaggle https://www.kaggle.com/kkkkkkk880/similar-sentenves-sbert

import torch
import pandas as pd
from sentence_transformers import evaluation,losses, SentenceTransformer, util, models, InputExample
from torch.utils.data import DataLoader
# http://git.nlp.ipipan.waw.pl/Scwad/SCWAD-CDSCorpus/tree/master/CDSCorpus
corpus = pd.read_csv('CDSCorpus/CDS_train.csv',sep='\t',error_bad_lines=False, encoding='utf-8')# ,nrows=1000  
corpus['relatedness_score'] = corpus['relatedness_score'].div(5)
corpus_test = pd.read_csv('CDSCorpus/CDS_test.csv',sep='\t',error_bad_lines=False, encoding='utf-8')  
corpus_test['relatedness_score'] = corpus_test['relatedness_score'].div(5)
# label2int = {"CONTRADICTION": 0, "ENTAILMENT": 1, "NEUTRAL": 2}
s1=[]
s2=[]
sc=[]
s3=[]
for index, row in corpus_test.iterrows():
    s1.append(row['sentence_A'])
    s2.append(row['sentence_B'])
    sc.append(row['relatedness_score'])
    # sc.append(label2int[row['entailment_judgment']])

evaluator = evaluation.EmbeddingSimilarityEvaluator(s1, s2, sc)# then change to corpus_test data


# roberta_large requires more gpu memory
word_embedding_model = models.Transformer('roberta_base', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_examples = []
test_examples = []
for index, row in corpus.iterrows():
    train_examples.append(InputExample(texts=[row['sentence_A'], row['sentence_B']], label=row['relatedness_score'])) 
    s3.append(row['sentence_A'])
    s3.append(row['sentence_B'])
    
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=5, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)

model.best_score #1epochs 0.9232 #5epochs 0.932
model.evaluate(evaluator) #0.9232
# model.save("roberta_base_CDS_train_biencoder")

# -----------------------------------------------
# from sentence_transformers import CrossEncoder
# model = CrossEncoder('roberta_base', max_length=256)
# model.fit(train_dataloader,
#           epochs=1, warmup_steps=100)

# scores = model.predict([[sentences1,sentences2 ],[sentences3,sentences2],[sentences1,sentences3 ]])
# #pretrained model 0.48104742, 0.48180264, 0.47577295
# #after training 0.26556703, 0.03470451, 0.03307376
# --------------------------------------------

sentences1 = 'Piłka nożna z wieloma grającymi facetami'
sentences2 = 'Jacyś mężczyźni grają w futbol'
sentences3 = 'Kobiety idą do fryzjera'

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
embeddings3 = model.encode(sentences3, convert_to_tensor=True)

#Compute cosine-similarits                       pretrained model | after training
print(util.pytorch_cos_sim(embeddings1, embeddings2).item())# 0.90 0.55 
print(util.pytorch_cos_sim(embeddings3, embeddings2).item())# 0.86 0.053 
print(util.pytorch_cos_sim(embeddings1, embeddings3).item())# 0.83 0.020 

# semantic_search
corpus = ["Mężczyzna je jedzenie. ","Mężczyzna je kawałek chleba.","Dziewczynka niesie dziecko.","Mężczyzna jedzie na koniu.","Kobieta gra na skrzypcach.","Dwóch mężczyzn pcha wózek przez las.","Mężczyzna jedzie na białym koniu po zamkniętym terenie.""Małpa gra na bębnach.","Gepard biegnie za swoją ofiarą."]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
s3=list(set(s3))
corpus_embeddings2 = model.encode(s3, convert_to_tensor=True)
queries = ['Mężczyzna je makaron.', 'Ktoś w kostiumie goryla gra na zestawie perkusyjnym.', 'Gepard goni ofiarę po polu.']

top_k = min(5, len(s3)) #top_k = min(5, len(corpus))
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings2)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    # for score, idx in zip(top_results[0], top_results[1]):
    #     print(s3[idx], "(Score: {:.4f})".format(score))

    hits = util.semantic_search(query_embedding, corpus_embeddings2, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(s3[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


# Paraphrase Mining - finding texts with similar meaning for large colections of sentences 10000+
largecorpus =corpus_test['sentence_A'].unique()
paraphrases = util.paraphrase_mining(model,largecorpus)

df = pd.DataFrame.from_records(paraphrases)
df[1] = [largecorpus[idx] for idx in df[1]] 
df[2] = [largecorpus[idx] for idx in df[2]] 
# df.to_csv("Paraphrase_Mining_pl.csv",index=False,header=["score","sentence1","sentence2"])


import spacy
nlp = spacy.load("pl_core_news_lg")

s1 = nlp(sentences1)
s2 = nlp(sentences2)
s3 = nlp(sentences3)
# sentence similarity
print(s1.similarity(s2)) 
print(s3.similarity(s2))
print(s1.similarity(s3)) 

# sentence embeddings
apple1.vector  # or apple1.tensor.sum(axis=0)







#################### cosine similarity
# Query: Mężczyzna je makaron.

# Top 5 most similar sentences in corpus:
# Mężczyzna je jedzenie.  (Score: 0.7490)
# Mężczyzna je kawałek chleba. (Score: 0.7456)
# Mężczyzna jedzie na koniu. (Score: 0.4072)
# Mężczyzna jedzie na białym koniu po zamkniętym terenie.Małpa gra na bębnach. (Score: 0.2964)
# Gepard biegnie za swoją ofiarą. (Score: 0.2015)


# ======================


# Query: Ktoś w kostiumie goryla gra na zestawie perkusyjnym.

# Top 5 most similar sentences in corpus:
# Kobieta gra na skrzypcach. (Score: 0.3038)
# Mężczyzna jedzie na białym koniu po zamkniętym terenie.Małpa gra na bębnach. (Score: 0.2279)
# Mężczyzna jedzie na koniu. (Score: 0.1826)
# Mężczyzna je jedzenie.  (Score: 0.1572)
# Gepard biegnie za swoją ofiarą. (Score: 0.1445)


# ======================


# Query: Gepard goni ofiarę po polu.

# Top 5 most similar sentences in corpus:
# Gepard biegnie za swoją ofiarą. (Score: 0.8232)
# Mężczyzna jedzie na białym koniu po zamkniętym terenie.Małpa gra na bębnach. (Score: 0.3133)
# Mężczyzna jedzie na koniu. (Score: 0.2899)
# Dwóch mężczyzn pcha wózek przez las. (Score: 0.1739)
# Mężczyzna je jedzenie.  (Score: 0.1574)

# ======================


# Query: Mężczyzna je makaron.

# Top 5 most similar sentences in corpus:
# Mężczyzna leży . (Score: 0.9562)
# Mężczyzna jedzie . (Score: 0.9538)
# Mężczyzna biegnie . (Score: 0.9532)
# Mężczyzna pływa . (Score: 0.9530)
# Mężczyzna jedzie skuterem . (Score: 0.9529)


# ======================


# Query: Ktoś w kostiumie goryla gra na zestawie perkusyjnym.

# Top 5 most similar sentences in corpus:
# Na chodniku pod budynkiem mężczyzna z maską na twarzy gra na bębnach , a trzymający rybę w rękach mężczyzna gra na gitarze . (Score: 0.9627)
# Mężczyzna w białym kapeluszu gra na saksofonie obok żółtego hydrantu . (Score: 0.9625)
# Na chodniku pod budynkiem mężczyzna z maską na twarzy gra na bębnach , a stojący obok mężczyzna gra na gitarze . (Score: 0.9622)
# Człowiek w czarnym kombinezonie gra na gitarze . (Score: 0.9620)
# Trzy dziewczynki obserwują mężczyznę grającego na gitarze i chłopca grającego na harmonijce ustnej . (Score: 0.9620)


# ======================


# Query: Gepard goni ofiarę po polu.

# Top 5 most similar sentences in corpus:
# Mały pies goni na trawie tenisową piłkę . (Score: 0.9470)
# Mały pies goni na trawie czerwoną piłkę . (Score: 0.9446)
# Ludzie biegną w wyścigu nocą . (Score: 0.9429)
# Człowiek jedzie konno w pobliżu krzewów . (Score: 0.9420)
# Delfin wystawia głowę i płetwy . (Score: 0.9411)


# # distinct classes !!! not working/predicting
# for index, row in corpus_test.iterrows():
#     test_examples.append(InputExample(texts=[row['sentence_A'], row['sentence_B']], label=label2int[row['entailment_judgment']])) 
    
# for index, row in corpus.iterrows():
#     train_examples.append(InputExample(texts=[row['sentence_A'], row['sentence_B']], label=label2int[row['entailment_judgment']])) 

# from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
# from sentence_transformers import CrossEncoder
# model = CrossEncoder('roberta_base', max_length=512, num_labels=3)
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
# evaluator =CEBinaryAccuracyEvaluator.from_input_examples(test_examples)
# model.fit(train_dataloader,epochs=1, warmup_steps=100,evaluator=evaluator,evaluation_steps=1000)
# model.predict([corpus_test["sentence_A"],corpus_test["sentence_B"]])
# corpus_test["predictions"] = corpus_test[["sentence_A", "sentence_B"]].apply(
#     lambda s: model.predict([s[0],s[1]]), axis=0
# )

# model.predict(["Żaden mężczyzna nie stoi na przystanku autobusowym .",
# "Mężczyzna z żółtą i białą reklamówką w ręce stoi na przystanku obok autobusu ."])