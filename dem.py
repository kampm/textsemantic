# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:25:02 2020

@author: Kamil
"""


import requests

x = requests.get('https://harambelogs.pl/channel/demonzz1/2020/08/20')
x.encoding
for i in range(8,13,1):
    y=list()
    for j in range(1,32,1):
        # print (f' {i}  {j}')
        x = requests.get(f'https://harambelogs.pl/channel/demonzz1/2020/{i}/{j}')
        if x.status_code == 200:
            for a in x.text.split("\n"):
                
                y.append(a.replace(' #demonzz1 ', ' '))
        else:
            print (f'blad {i}  {j}')
            
        print (f'y {i}  {j}')
    print (f'x {i}  {j}')
    ypd=pd.DataFrame(y)
    ypd.to_csv(f'dem_{i}.csv',index=False,encoding="utf-8-sig")


import glob
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# df = pd.concat(map(pd.read_csv, glob.glob('dem_*.csv')))

df100=pd.read_csv('demall.csv')
df100=df100.drop(df100.columns[0], axis=1)
df100 = df100.dropna(axis=0, subset=['msg'])


# df100['time']=df100['0'].str.split('] ',n=-1).str[0].str.replace('[','')
# df100['user']=df100['0'].str.split('] ',n=-1).str[1].str.split(': ',n=-1).str[0]
# df100['msg']=df100['0'].str.split('] ',n=-1).str[1].str.split(': ',n=-1).str[1]

# df100.to_csv('demall.csv',encoding="utf-8-sig")

unique_users=df100.user.value_counts()
unique_users=unique_users[unique_users>10]
df100.head(50)
df100['time']=pd.to_datetime(df100['time'],format='%Y-%m-%d %H:%M:%S')
def MesFromDateRange(df,start,end):
    return df.loc[(df['time']>start)&(df['time']<=end)]
    
unique_users=MesFromDateRange(df100,'2020-10-01','2020-11-01').user.value_counts()
unique_users.describe()
n=df100.shape[0]

# t='https://www.youtube.com/?gl=PL&hl=pl yt cos tam'
# t=t.lower()
# t=tokenizer.tokenize(t)
# t = list(filter(lambda tt: not tt.startswith('http'), t))

def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        # tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'
    
def postprocess(data, n=n):
    data = data.head(n)
    data['tokens'] = data['msg'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(df100)

data.to_csv('demdata.csv.gz',compression='gzip')

# n=100
# x_train, x_test = train_test_split(np.array(data.head(n).tokens), test_size=0.0)
x_train=np.array(data.head(n).tokens)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        t=list()
        for a in v:
            t.append(a)
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(t, [label]))
    # print(labelized)
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
# x_test = labelizeTweets(x_test, 'TEST')

x_train[0]
# t=x_train[1]
# print(x_train[0][0])

# for i in x_train[0][0]: 
#     x_train[0][0]=t.append(i)
#     print(i)

n_dim=300
tweet_w2v = Word2Vec(size=n_dim, min_count=5)
tweet_w2v.build_vocab([x[0] for x in tqdm(x_train)])
# print([x[0] for x in tqdm(x_train)])
tweet_w2v.train([x[0] for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)

tweet_w2v.wv.index2entity[:100]
tweet_w2v.save("dem_ndim300_min5.model")
#tweet_w2v= Word2Vec.load("dem_ndim300_min5.model")
print(tweet_w2v['me'])

print(tweet_w2v.most_similar('hitleronzz'))

##word mistakes
import nltk
spell_mistake_min_frequency = 3
fasttext_min_similarity = 0.6
def include_spell_mistake(word, similar_word, score):
    # print(score, )
    edit_distance_threshold = 1 if len(word) <= 4 else 2
    return (
        score > fasttext_min_similarity and 
        len(similar_word) > 3
            and vocab[similar_word] >= spell_mistake_min_frequency
            # and not enchant_us.check(similar_word)
            # and word[0] == similar_word[0]
            # and nltk.edit_distance(word, similar_word) <= edit_distance_threshold
            )

df100['msg'].value_counts().head()
vocab = collections.Counter(df100['msg'].str.lower())
vocab.most_common(10)
word_to_mistakes = collections.defaultdict(list)
nonalphabetic = re.compile(r'[^a-zA-Z]')
 
for word, freq in vocab.items():
    # print(word,freq)
    if freq < 50 or len(word) <= 3 or nonalphabetic.search(word) is not None:

        continue
 
    # Query the fasttext model for 50 closest neighbors to the word
    similar_words = tweet_w2v.wv.most_similar(word, topn=50)
    for similar_word in similar_words:
        # print(include_spell_mistake(word, similar_word[0], similar_word[1]))
        if include_spell_mistake(word, similar_word[0], similar_word[1]):
            word_to_mistakes[word].append(similar_word)



print(list(word_to_mistakes.items())[:10])


#####html 2d tsne
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show

# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0,n_jobs=-1)
tsne_w2v = tsne_model.fit_transform(word_vectors)
np.save('tsne_w2v.npy',tsne_w2v)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())

tsne_df[tsne_df.eq('0097').any(1)]
# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
output_file("foo_ndim300min5allvocab.html")
show(plot_tfidf)
