from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import clusters_mg
import numpy as np
import sys

def cluster(titles, data, n, filename):

    clust=clusters_mg.kcluster(data,distance=clusters_mg.cosine,k=n)
    mat = []
    for i in range(len(clust)):
        mat.append([titles[r] for r in clust[i]])
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for i in range(len(clust)):
            print('cluster '+ str(i) + ':\n' + str([titles[r] for r in clust[i]]))
    sys.stdout = original_stdout
    cor(data, clust)
    return  mat, clust

def getWords(clust, data):
    t = []
    for i in clust:
        words = []
        for j in i:
            words.extend(data[j])
       # print(words)
        t.append(" ".join(words))
    original_stdout = sys.stdout
    with open('cwords.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for i in t:
            print(i)
            print("\n")
    sys.stdout = original_stdout

def cor(data, clust):
   # print(clust)
    # Is there any correlation between features?
    mat = []
    flat_clust = [item for sublist in clust for item in sublist]
    data = [data[i] for i in flat_clust]
    data1 = data[:300]
    #print(data)
    for i in data1:
        row = []
        for j in data:
           #print(i,j)
          # print(type(i),type(j))
           row.append(clusters_mg.cosine(i,j))
        mat.append(row)

    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(25, 25))
    #corr = mat.corr()
    svm = sns.heatmap(mat, center=0,annot=False,  linewidths=.6, linecolor = 'white',xticklabels = False, yticklabels = False, ax=ax)
    plt.show()

    mat = []
    data2 = data[300:]
    # print(data)
    for i in data2:
        row = []
        for j in data:
            row.append(clusters_mg.cosine(i, j))
        mat.append(row)

    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(25, 25))
    # corr = mat.corr()
    svm = sns.heatmap(mat, center=0, annot=False, linewidths=.6, linecolor='white', xticklabels=False,
                      yticklabels=False, ax=ax)
    plt.show()
    #
    # mat = []
    # data3 = data[400:]
    # # print(data)
    # for i in data3:
    #     row = []
    #     for j in data:
    #         row.append(clusters_mg.cosine(i, j))
    #     mat.append(row)
    #
    # mat = np.array(mat)
    # fig, ax = plt.subplots(figsize=(25, 25))
    # # corr = mat.corr()
    # svm = sns.heatmap(mat, center=0, annot=False, linewidths=.7, linecolor='white', xticklabels=False,
    #                   yticklabels=False, ax=ax)
    # plt.show()
    # #figure = svm.get_figure()
    # mat = []
    # data4 = data[:300]
    # # print(data)
    # for i in data4:
    #     row = []
    #     for j in data:
    #         row.append(clusters_mg.cosine(i, j))
    #     mat.append(row)
    #
    # mat = np.array(mat)
    # fig, ax = plt.subplots(figsize=(25, 25))
    # # corr = mat.corr()
    # svm = sns.heatmap(mat, center=0, annot=False, linewidths=.7, linecolor='white', xticklabels=False,
    #                   yticklabels=False, ax=ax)
    # plt.show()