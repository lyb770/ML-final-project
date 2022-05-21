from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from lda_utils_mg import *


def keys_to_counts(keys):
    '''returns a tuple of topic categories and their accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)
import matplotlib.mlab as mlab
import seaborn as sb
import ast

import matplotlib.pyplot as plt

import pandas as pd
from IPython.display import display
#from jedi.refactoring import inline
from tqdm import tqdm


from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from sklearn.manifold import TSNE


def plot_top_words(reindexed_data):
    count_vectorizer = TfidfVectorizer()
    words, word_values = get_top_n_words1(n_top_words=10, count_vectorizer=count_vectorizer, text_data=reindexed_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(words)), word_values)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words)
    ax.set_title('Top Words')

    plt.show()

def get_topics_distribution(topic_matrix):
    return topic_matrix.tolist()


def get_best_topic_perdoc(topic_matrix):
    t = get_topics_distribution(topic_matrix)
    for i in range(len (t)):
        lst = t[i]
        indx = 0
        max_val = 0
        for j in range(len(lst)):
            if lst[j] > max_val:
                max_val = lst[j]
                indx = j
        t[i] = indx

    return t


def extract_lda_model(lda_topic_matrix, document_term_matrix, n_topics,
                      count_vectorizer,
                      show_plot = False, show_cluster = False):
   # lda_model = LatentDirichletAllocation(n_components=n_topics, learning_decay = .5, max_iter=50,
                                         # random_state=10, verbose=0)
    #lda_topic_matrix = lda_model.fit_transform(document_term_matrix)

    lda_keys = get_keys(lda_topic_matrix)

    lda_categories, lda_counts = keys_to_counts(lda_keys)

    top_n_words_lda = get_top_n_words(20, lda_keys, document_term_matrix, count_vectorizer, n_topics)

    print("\nLDA topic words")
    for i in range(len(top_n_words_lda)):
        topic_word_list = top_n_words_lda[i].split()

        print(topic_word_list)

    if show_plot:
       # top_3_words = get_top_n_words(3, lda_keys, document_term_matrix, count_vectorizer, n_topics)

       #labels = [top_3_words[i] for i in range(len(lda_categories))]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(lda_categories, lda_counts)
        ax.set_xticks(lda_categories)
       # ax.set_xticklabels(labels)
        ax.set_title('LDA Topic Category Counts')
        plt.show()
    if show_cluster:
        colormap = np.array([
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
            "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
            "#27AE60", "#F1C40F", "#3498DB", "#154360", "#4A235A",
            "#7B7D7D", "#641E16", "#17202A", "#F5B041", "#58D68D",
            "#F1948A", "#D2B4DE", "#EDBB99", "#D5DBDB", "#2C3E50 ",
            "#6E2C00", "#21618C", "#F9E79F", "#A9DFBF", "#212F3D "
        ])
        colormap = colormap[:n_topics]
        tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100,
                          n_iter=2000, verbose=1, random_state=0, angle=0.75)
        tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)

        #top_3_words_lda = get_top_n_words(3, lda_keys, document_term_matrix, count_vectorizer, n_topics)
        #  text=top_3_words_lda[t],
        lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors, len(tsne_lda_vectors))
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=700, plot_height=700)
        plot.scatter(x=tsne_lda_vectors[:, 0], y=tsne_lda_vectors[:, 1], color=colormap[lda_keys])

        for t in range(len(lda_mean_topic_vectors)):
            label = Label(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1],
                         text_color=colormap[t])
            plot.add_layout(label)

        show(plot)

    return lda_topic_matrix


def make(document_term_matrix, count_vectorizer, n_topics):
    #count_vectorizer =CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    #count_vectorizer = CountVectorizer(tokenizer=lambda doc: doc,min_df=5,max_features=1000,binary=True )
    # test = get_document_term_matrix_asarray(reindexed_data, reindexed_data.index)
    #document_term_matrix = get_document_term_matrix(data, count_vectorizer)

    # lsa_model(document_term_matrix, n_topics, count_vectorizer )
    lda_topic_matrix = extract_lda_model(document_term_matrix, n_topics,
                                         count_vectorizer, True,True)
    document_topics = get_best_topic_perdoc(lda_topic_matrix)
