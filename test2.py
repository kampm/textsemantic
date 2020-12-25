# -*- coding: utf-8 -*-
# https://miroslawmamczur.pl/jak-dziala-metoda-redukcji-wymiarow-t-sne/
import pandas as pd
import fasttext
from gensim.models import KeyedVectors, FastText
import matplotlib.pyplot as plt
import numpy as np
# data = pd.read_csv('C:/Users/Kamil/Downloads/t.vec',quoting=3,
#                         error_bad_lines=False)

input_data=10000
%%time
data = KeyedVectors.load_word2vec_format("C:/Users/Kamil/Downloads/t.vec",limit=input_data)

len(list(data.vocab)[0])

data.most_similar("Leszek")
data.similarity("kot","pies")


##
word_vectors = data.wv
wanted_words = []
count = 0
for word in word_vectors.vocab:
    if count<input_data:
        wanted_words.append(word)
        count += 1
    else:
        break
wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)
wanted_vocab

X = data[wanted_vocab] # X is an array of word vectors, each vector containing 150 tokens
# from tsnecuda import TSNE
# from openTSNE import TSNE
# %%time
# tsne_model = TSNE(perplexity=40, n_components=2,verbose=1, initialization='pca', n_iter=1000, random_state=42,n_jobs=-1)
# Y = tsne_model.fit(X)


from sklearn.manifold import TSNE
# %%time
tsne_model = TSNE(perplexity=40, n_components=2,verbose=1, init="pca", n_iter=1000, random_state=42,n_jobs=-1)
Y = tsne_model.fit_transform(X)
# Y=np.load('tsne_FT_1M.npy')
np.save('tsne_FT_100k.npy',Y)

#Plot the t-SNE output
fig, ax = plt.subplots(figsize=(200,200))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
ax.set_yticklabels([]) #Hide ticks
ax.set_xticklabels([]) #Hide ticks
_ = plt.show()

###3 dim
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(Y[:, 0], Y[:, 1],
           #  Y[:, 2],
           cmap=plt.cm.spring, edgecolor='k', s=130)
ax.set_title("First three PCA components")
ax.set_xlabel("1st PCA component")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd PCA component")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd PCA component")
ax.w_zaxis.set_ticklabels([])

plt.show()
###
#####html 2d

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, show

# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="pl",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
# word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())]

# dimensionality reduction. converting the vectors to 2d vectors
# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0,n_jobs=-1)
# tsne_w2v = tsne_model.fit_transform(word_vectors)
# np.save('tsne_w2v.npy',tsne_w2v)

# putting everything in a dataframe
tsne_df = pd.DataFrame(Y, columns=['x', 'y'])
tsne_df['words'] = list(data.wv.vocab.keys())

tsne_df[tsne_df.eq('Robert').any(1)]
# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
output_file("test2_1.html")
show(plot_tfidf)

#####

####tensorboard
# tensorboard --logdir F:\PycharmProjects\zajecia\spyder\semantic\project-tensorboard\log-1
import os
import tensorflow as tf
from tensorboard.plugins import projector

PATH = os.getcwd()
LOG_DIR = PATH + '/project-tensorboard/log-1/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    

# Save Labels separately on a line-by-line manner.
with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w", encoding="utf-8") as f:
  for word in words:
    f.write("{}\n".format(word))
  # Fill in the rest of the labels with "unknown"
  # for unknown in range(1, encoder.vocab_size - len(encoder.subwords)):
  #   f.write("unknown #{}\n".format(unknown))
    
tf_data = tf.Variable(X)
checkpoint = tf.train.Checkpoint(embedding=tf_data)
checkpoint.save(os.path.join(LOG_DIR, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(LOG_DIR, config)

####

