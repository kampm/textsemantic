# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:46:37 2020

@author: Kamil
"""

import itertools
from mittens import Mittens,GloVe
from gensim.models import KeyedVectors
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool,Legend, LegendItem
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show
from bokeh.palettes import Category10_3,Category20_20 ,Category10_10
from bokeh.transform import factor_cmap, factor_mark
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
#https://github.com/sdadas/polish-nlp-resources

n_dim=300

# Load `cooccurrence`
# Train GloVe model
# glove_model = GloVe(n=n_dim, max_iter=1000)  # 25 is the embedding dimension

# embeddings = glove_model.fit(x_train)

####
word2vec = KeyedVectors.load_word2vec_format("glove_100_3_polish.txt", limit=1000)
model=word2vec

def kmean(word2vec,tsne_df,n_clus):    
    from sklearn.cluster import KMeans
    kmeans = KMeans(init='k-means++', n_clusters=n_clus, n_init=10,max_iter=2000,n_jobs=-1,random_state=42)
    kmeans.fit(word2vec.vectors)
    y_kmeans = kmeans.predict(word2vec.vectors)
    tsne_df['categ'] = list(y_kmeans)
    tsne_df['categ'] =tsne_df['categ'].map(str)

def plt2html(tsne_w2v,word2vec,htmlname):
    #####html 2d tsne
    # defining the chart
    output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = list(word2vec.wv.vocab.keys())
    
    kmean(word2vec,tsne_df,n_clus)
    
    # tsne_df[tsne_df.eq('0097').any(1)]

    # MARKERS = ['x', 'circle', 'triangle']
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=tsne_df,
                  color=factor_cmap('categ', Category20_20 , categ),legend='categ')
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    output_file(htmlname)
    show(plot_tfidf)

print(word2vec.similar_by_word("bierut"))


# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [word2vec[w] for w in list(word2vec.wv.vocab.keys())]

# dimensionality reduction. converting the vectors to 2d vectors
tsne_model = TSNE(n_components=2, verbose=1, random_state=0,n_jobs=-1)
tsne_w2v = tsne_model.fit_transform(word_vectors)
# np.save('tsne_glove10k.npy',tsne_w2v)


n_clus=15
categ = list(map(str, range(0,n_clus,1)))

plt2html(tsne_w2v, word2vec, "golvepltest10k100dim.html")


##test kmeans

plt.scatter(tsne_w2v[:, 0], tsne_w2v[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
