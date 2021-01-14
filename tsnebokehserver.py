# -*- coding: utf-8 -*-
"""
env cmd 
bokeh serve --show F:\PycharmProjects\zajecia\spyder\semantic\textsemantic\tsnebokehserver.py
"""

from bokeh.plotting import curdoc, figure
from sklearn.cluster import KMeans
import sys
# import ncvis
from fast_tsne import fast_tsne
import itertools
# from mittens import Mittens, GloVe
from gensim.models import KeyedVectors
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, Legend, LegendItem, Panel, ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show
from bokeh.palettes import Category10_3, Category20_20, Category10_10,Turbo256 
from bokeh.transform import factor_cmap, factor_mark
from bokeh.application.handlers import FunctionHandler
from bokeh.models.widgets import CheckboxGroup, Slider, RangeSlider, Tabs,Button
from bokeh.layouts import column, row, WidgetBox
from bokeh.application import Application
# from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import random
import seaborn as sns
import os
os.chdir('F:\\PycharmProjects\\zajecia\\spyder\\semantic\\textsemantic\\')
sns.set_style('ticks')
sys.path.append(
    'F:\\PycharmProjects\\zajecia\\spyder\\semantic\\textsemantic\\FIt-SNE\\')  # FIt-SNE exe path


n_clus = 40  # number of clusters
categ = list(map(str, range(0, n_clus, 1)))


def make_plot(src):

    p = figure(plot_width=700, plot_height=600, title="NLP TSNE",
               tools="pan,wheel_zoom,box_zoom,reset,hover,save",
               x_axis_type=None, y_axis_type=None, min_border=1)

    p.scatter(x='x', y='y', source=src,
              color=factor_cmap('categ',random.sample(Turbo256,n_clus), categ), legend='categ')
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = {"word": "@words"}

    return p


def make_dataset(carrier_list):
    tab = tsne_df.loc[tsne_df['categ'].isin(carrier_list)]
    return ColumnDataSource(tab)




print("Loading file")
word2vec = KeyedVectors.load_word2vec_format(
    "glove_100_3_polish.txt", limit=10000)
print("Loading complete")

word_vectors = [word2vec[w] for w in list(word2vec.index_to_key)]

perplexity=50
early_exag_coeff=50

FIt_tsne_model = fast_tsne(word_vectors, perplexity=perplexity,
                           early_exag_coeff=early_exag_coeff, seed=42)


tsne_df = pd.DataFrame(FIt_tsne_model, columns=['x', 'y'])

print("Kmeans..")
kmeans = KMeans(init='k-means++', n_clusters=n_clus,
                n_init=10, max_iter=300, n_jobs=-1, random_state=42)

vec = word2vec.vectors  # all dimensions
# vec = tsne_df           #2 dimensions
kmeans.fit(vec)
y_kmeans = kmeans.predict(vec)
print("Kmeans done")

tsne_df['categ'] = list(y_kmeans)
tsne_df['categ'] = tsne_df['categ'].map(str)
print("Score: ", kmeans.score(vec))

tsne_df['words'] = list(word2vec.index_to_key)

doc = curdoc()
# tsne_df.head()
carrier_selection = CheckboxGroup(labels=list(
            tsne_df["categ"].unique()), active=list(range(len(tsne_df["categ"].unique()))))


def update(attr, old, new):
    # Get the list of carriers for the graph

    global n_clus,FIt_tsne_model,perplexity ,early_exag_coeff,carrier_selection, initial_carriers, src, p, layout, tab, tabs,select_all, categ, kmeans, vec, y_kmeans, tsne_df
    carriers_to_plot = [carrier_selection.labels[i] for i in
                        carrier_selection.active]

    bin_width = binwidth_select.value
    perp_value = perplexity_select.value
    exag_value = early_exag_coeff_select.value
    if((n_clus != bin_width) or (perplexity != perp_value) or (exag_value != early_exag_coeff)):
        if((perplexity != perp_value) or (exag_value != early_exag_coeff)):
            early_exag_coeff = exag_value
            perplexity = perp_value
            FIt_tsne_model = fast_tsne(word_vectors, perplexity=perplexity,
                                       early_exag_coeff=early_exag_coeff, seed=42)
            tsne_df = pd.DataFrame(FIt_tsne_model, columns=['x', 'y'])

        doc.remove_root(tabs)
        n_clus = bin_width  # number of clusters
        categ = list(map(str, range(0, n_clus, 1)))
        tsne_df = pd.DataFrame(FIt_tsne_model, columns=['x', 'y'])
        print("Kmeans..")
        kmeans = KMeans(init='k-means++', n_clusters=n_clus,
                        n_init=10, max_iter=2000, n_jobs=-1, random_state=42)
        vec = word2vec.vectors  # all dimensions
        # vec = tsne_df           #2 dimensions
        kmeans.fit(vec)
        y_kmeans = kmeans.predict(vec)
        print("Kmeans done")
        tsne_df['categ'] = list(y_kmeans)
        tsne_df['categ'] = tsne_df['categ'].map(str)
        tsne_df['words'] = list(word2vec.index_to_key)
        carrier_selection = CheckboxGroup(labels=list(
            tsne_df["categ"].unique()), active=list(range(len(tsne_df["categ"].unique()))))
        controls = WidgetBox(binwidth_select,perplexity_select,early_exag_coeff_select,select_all,carrier_selection)
        initial_carriers = [carrier_selection.labels[i]
                            for i in carrier_selection.active]
        carrier_selection.on_change('active', update)
        carriers_to_plot = [carrier_selection.labels[i] for i in
                            carrier_selection.active]
        src = make_dataset(initial_carriers)
        p = make_plot(src)
        layout = row(controls, p)
        tab = Panel(child=layout, title='Kmeans categories')
        tabs = Tabs(tabs=[tab])
        doc.add_root(tabs)
    # Make a new dataset based on the selected carriers and the
    # make_dataset function defined earlier
    new_src = make_dataset(carriers_to_plot)
    src.data.update(new_src.data)

    # Convert dataframe to column data source
    # new_src = ColumnDataSource(new_src)

    print(new_src)
    # Update the source used the quad glpyhs


binwidth_select = Slider(start = 5, end = 100, 
                     step = 1, value = 40,
                     title = 'Clusters')
perplexity_select = Slider(start = 5, end = 500, 
                     step = 1, value = 50,
                     title = 'perplexity')
early_exag_coeff_select = Slider(start = 5, end = 500, 
                     step = 1, value = 50,
                     title = 'early_exag_coeff')


select_all = Button(label="unselect all")
def unselect():
    carrier_selection.active = []
select_all.on_click(unselect)

controls = WidgetBox(binwidth_select,perplexity_select,early_exag_coeff_select,select_all,carrier_selection)

initial_carriers = [carrier_selection.labels[i]
                    for i in carrier_selection.active]

carrier_selection.on_change('active', update)
binwidth_select.on_change('value_throttled', update)
perplexity_select.on_change('value_throttled', update)
early_exag_coeff_select.on_change('value_throttled', update)

src = make_dataset(initial_carriers)

p = make_plot(src)

layout = row(controls, p)

tab = Panel(child=layout, title='Kmeans categories')
tabs = Tabs(tabs=[tab])

doc.add_root(tabs)

