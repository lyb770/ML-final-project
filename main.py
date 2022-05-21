import math
import numpy as np
import pandas as pd
import sys
import json
import pickle
import string
from Classifier import rec
from Topic_modle import vec, LDA
from Clean import good_words
from Vis import make
import kmcluster
import kmcluster_mg
import corr_vis
import Conf_mat
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
def getTitles():
    filename = "cornell movie-dialogs corpus//movie_titles_metadata.txt"
    temp = []
    docs = []
    with open(filename) as file:
        for line in file:
            temp.append(line.rstrip().split("+++$+++"))
        # print(temp)
    lines = []
    for i in range(len(temp)):
        a = temp[i][1].strip().rstrip().lower()
        docs.append(a)
    return docs


def load():
    f = open("jwords")
    data = list(json.load(f))
    #print(data[0])
    #print(data[len(data)-1])
    return data
def j():

    f = open("jdata")
    data = list(json.load(f))

    users = []
    for i in data:
        a = []
        b = []
        for j in i:
            a.append(j[0])
            b.append(j[1])
        users.append((a,b))
    return users


def main():
   titles = getTitles()
   movies = j()
   clean_corpus = good_words()

   cv_arr, vocab_cv, vector, tf_idf_arr, vocab_tf_idf = vec(clean_corpus)
   atr, non = LDA(cv_arr, vocab_cv,vector,5)
   tfcount = []

   mtf = tf_idf_arr.todense()

   for k in mtf:
       tfcount.append(np.asarray(k)[0])
   kmcluster_mg.cluster(titles, tfcount, 5,'cTFclusters.txt')
   kmcluster.cluster(np.array(titles), tfcount, 5, 'TFclusters.txt')

   all = []

   for i in range(617) :
       all.append(np.append(tfcount[i],atr[i]))
   kmcluster_mg.cluster(titles, all, 5, 'cAllclusters.txt')
   kmcluster.cluster(np.array(titles),all,5, 'Allclusters.txt')

   corr_vis.cor(atr)
   ran = list(range(len(titles)))
   doc_topic = [titles[i] for i in ran if i not in non]
   clust, index = kmcluster_mg.cluster(titles, atr,5,'clusters.txt')
   Conf_mat.make(clust)
   kmcluster.cluster(np.array(titles), atr, 5, 'Tclusters.txt')

if __name__ == '__main__':
    main()


