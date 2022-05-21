from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np


def rec(movies, atr, n):
    kte = []
    ktr = []
    num = []

    for i in movies:
        if len(i[0]) > 150:
            vector = []
            lable = []
            m_user = i[0]
            for j in m_user:
                if j == 616:
                    vector.append([100000] * n)
                else:
                    vector.append((atr[j].tolist()))
            for k in i[1]:
                if k < 3.5:
                    lable.append('0')
                else:
                    lable.append('1')

            knn_cv = KNeighborsClassifier(n_neighbors=9, weights='distance', metric="manhattan")
            cv_scores = cross_val_score(knn_cv, vector, lable, cv=10)
            kte.append(np.mean(cv_scores))
            num.append(len(lable))
    fig = plt.figure()
    plt.scatter(kte, num)
    plt.title(str(n) + "topics")
    name = "category: plot " + str(n) + ".png"
    fig.savefig(name)
    plt.show()
