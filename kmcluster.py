from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import kmcluster_mg
import seaborn as sns
import clusters_mg
import numpy as np
import sys
def cluster(titles, data, n, filename):

    print ('2 clusters:')
    kmeans = KMeans(n_clusters=n, random_state=10).fit(data)
    clust = []
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        for i in range(n):
            filter = titles[np.array(kmeans.labels_) == i]
            index = [np.array(kmeans.labels_) == i]
            for j in filter:
                print(j+",")
            print("\n")
            clust.append(filter)
        print(kmeans.inertia_)
        sys.stdout = original_stdout


    kmcluster_mg.cor(data, index)

    return kmeans.inertia_,  clust
