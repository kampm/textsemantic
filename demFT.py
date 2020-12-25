# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:48:16 2020

@author: Kamil
"""

def download_data(channel_name):
    from tqdm import tqdm
    tqdm.pandas(desc="progress-bar")
    import requests
    import pandas as pd

    y=list()
    for i in tqdm(range(8,13,1)):
        for j in range(1,32,1):
            x = requests.get(f'https://harambelogs.pl/channel/{channel_name}/2020/{i}/{j}')
            if x.status_code == 200:
                for a in x.text.split("\n"):
                    y.append(a.replace(f' #{channel_name} ', ' '))
            else:
                continue
                # print (f'blad {i}  {j}')
                
    dataDF = pd.DataFrame(y)
    dataDF['time']=dataDF[0].str.split('] ',n=-1).str[0].str.replace('[','')
    dataDF['user']=dataDF[0].str.split('] ',n=-1).str[1].str.split(': ',n=-1).str[0]
    dataDF['msg']=dataDF[0].str.split('] ',n=-1).str[1].str.split(': ',n=-1).str[1]
    dataDF = dataDF.dropna(axis=0, subset=['msg'])
    dataDF = dataDF.drop(dataDF.columns[0], axis=1)
    print("-"*20)
    return dataDF

data = download_data("mamm0n")    

import gensim
import numpy as np
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.TaggedDocument
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

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
    
def postprocess(data, n):
    data = data.head(n)
    data['tokens'] = data['msg'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

datapp = postprocess(data,data.shape[0])

x_train=np.array(datapp.head(datapp.shape[0]).tokens)

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

n_dim=300
model = Word2Vec(size=n_dim, min_count=5)
model = FastText(size=n_dim, min_count=5)
model.build_vocab([x[0] for x in tqdm(x_train)])
model.train([x[0] for x in tqdm(x_train)],total_examples=model.corpus_count,epochs=model.iter)

model.wv.index2entity[:100]
# tweet_w2v.save("dem_ndim300_min5.model")
#tweet_w2v= Word2Vec.load("dem_ndim300_min5.model")
print(model['me'])

print(model.wv.most_similar('mówić'))

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
word_vectors = [model[w] for w in list(model.wv.vocab.keys())]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0,n_jobs=-1)
tsne_w2v = tsne_model.fit_transform(word_vectors)
# np.save('tsne_w2v.npy',tsne_w2v)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(model.wv.vocab.keys())

tsne_df[tsne_df.eq('0097').any(1)]
# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
output_file("mamw2v.html")
show(plot_tfidf)

#top words 
n_posts = 50
data['msg'].str.lower().value_counts().keys().values[:n_posts]
q_S = ' '.join(data['msg'].str.lower().value_counts().keys().values[:n_posts])
q_I = ' '.join(data['msg'].str.lower().value_counts().keys().values[:n_posts])

from wordcloud import WordCloud
from nltk import download
import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# download('stopwords')
stop = set(stopwords.words('polish.txt'))
wordcloud_S = WordCloud(max_font_size=None, stopwords=stop,scale = 2,colormap = 'Dark2').generate(q_S)
wordcloud_I = WordCloud(max_font_size=None, stopwords=stop,scale = 2,colormap = 'Dark2').generate(q_I)

fig, ax = plt.subplots(1,2, figsize=(80, 20))
ax[0].imshow(wordcloud_S)
ax[0].set_title('Top words sincere posts',fontsize = 20)
ax[0].axis("off")

ax[1].imshow(wordcloud_I)
ax[1].set_title('Top words INsincere posts',fontsize = 20)
ax[1].axis("off")

plt.show()


