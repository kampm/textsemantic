# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:46:38 2020

@author: Kamil
"""
# https://github.com/ksopyla/awesome-nlp-polish
# https://www.researchgate.net/publication/335130889_openTSNE_a_modular_Python_library_for_t-SNE_dimensionality_reduction_and_embedding
# https://arxiv.org/pdf/2001.11411.pdf
import sys
import ncvis
from fast_tsne import fast_tsne
import itertools
from mittens import Mittens, GloVe
from gensim.models import KeyedVectors
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, Legend, LegendItem, Panel,ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show
from bokeh.palettes import Category10_3, Category20_20, Category10_10
from bokeh.transform import factor_cmap, factor_mark
from bokeh.application.handlers import FunctionHandler
from bokeh.models.widgets import CheckboxGroup, Slider, RangeSlider, Tabs
from bokeh.layouts import column, row, WidgetBox
from bokeh.application import Application
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
#https://github.com/sdadas/polish-nlp-resources
import numpy as np
import pylab as plt
import seaborn as sns
sns.set_style('ticks')
sys.path.append(
    'F:\\PycharmProjects\\zajecia\\spyder\\semantic\\textsemantic\\FIt-SNE\\')

n_dim=300

# Load `cooccurrence`
# Train GloVe model
# glove_model = GloVe(n=n_dim, max_iter=1000)  # 25 is the embedding dimension

# embeddings = glove_model.fit(x_train)

####
word2vec = KeyedVectors.load_word2vec_format(
    "glove_100_3_polish.txt", limit=10000)
model = word2vec


def kmean(word2vec, tsne_df, n_clus):
    from sklearn.cluster import KMeans
    kmeans = KMeans(init='k-means++', n_clusters=n_clus,
                    n_init=10, max_iter=2000, n_jobs=-1, random_state=42)
    
    vec = word2vec.vectors  #all dimensions
    # vec = tsne_df           #2 dimensions
    kmeans.fit(vec) 
    y_kmeans = kmeans.predict(vec)
    tsne_df['categ'] = list(y_kmeans)
    tsne_df['categ'] = tsne_df['categ'].map(str)
    print ("Score: ", kmeans.score(vec))

def plt2html(tsne_w2v, word2vec, htmlname):
    # defining the chart
    output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A",
                           tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                           x_axis_type=None, y_axis_type=None, min_border=1)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    kmean(word2vec, tsne_df, n_clus)

    tsne_df['words'] = list(word2vec.index_to_key)

    plot_tfidf.scatter(x='x', y='y', source=tsne_df,
                       color=factor_cmap('categ', Category20_20, categ), legend='categ')
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"word": "@words"}
    output_file(htmlname)
    show(plot_tfidf)


# print(word2vec.similar_by_word("bierut"))


# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [word2vec[w] for w in list(word2vec.index_to_key)]

# # scikit-learn tsne 1,45h
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0,n_jobs=-1)
# %time tsne_w2v = tsne_model.fit_transform(word_vectors)
# # np.save('tsne_glove10k.npy',tsne_w2v)

# #ncvis 3min
# vis = ncvis.NCVis()
# %time  vis_tsne_model = vis.fit_transform(np.array(word_vectors))

#FIt-SNE 10min
%time FIt_tsne_model = fast_tsne(word_vectors,perplexity=50,early_exag_coeff=50,seed=42)


n_clus=40
categ = list(map(str, range(0,n_clus,1)))

plt2html(FIt_tsne_model, word2vec, "usunactest.html")

def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

plot_with_matplotlib(tsne_df['x'], tsne_df['y'],tsne_df['words'])


##test kmeans

plt.scatter(tsne_w2v[:, 0], tsne_w2v[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);





# output_file("tatat.html")
show(tabs)
