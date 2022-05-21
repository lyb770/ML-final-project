import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as od
import numpy as np
import clusters_mg
def dif(i,j):
    sum = 0
    for n in range(len(i)):
        sum+=i[n]-j[n]
    return abs(sum)
def cor(data):
    # Is there any correlation between features?
    mat = []

    data = data[0:300]

    for i in data:
        row = []
        for j in data:
           row.append(clusters_mg.cosine(i,j))
        mat.append(row)

    fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(mat, center=0, annot=False, linewidths=.4, linecolor='white', xticklabels=False,
                yticklabels=False, ax=ax)
    plt.show()
