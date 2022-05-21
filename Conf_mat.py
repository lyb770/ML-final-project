import ast

import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh


def getTitles():
    filename = "cornell movie-dialogs corpus//movie_titles_metadata.txt"
    temp = []
    titles = []
    genre = []
    with open(filename) as file:
        for line in file:
            temp.append(line.rstrip().split("+++$+++"))
        # print(temp)
    lines = []
    for i in range(len(temp)):
        a = temp[i][1].strip().rstrip().lower()
        b = temp[i][5].strip().rstrip().lower()
        x = ast.literal_eval(b)
        x = [n.strip() for n in x]
        titles.append(a)
        genre.append(x)
    return titles, genre


def topics():
    titles, genre = getTitles()
    flat_genre = [item for sublist in genre for item in sublist]
    flat_genre = np.array(flat_genre)
    flat_genre = np.unique(flat_genre)
    classes = []
    bad = []
    for i in flat_genre:
        count = 0
        temp = []
        for k in range(len(genre)):
            if (i in genre[k]):
                count+=1
                temp.append(titles[k])
        if count >= 20:
            classes.append(temp)
        else:
            bad.append(i)
    flat_genre = [i for i in flat_genre if i not in bad]
    return classes, flat_genre

def make(clust):
    #print(clust)
    classes, genre = topics()
    genre.append("Total")
    #print(classes)

    mat = []
    for k in clust:
        row = []
        for j in classes:
            count = 0
            for i in j:
                if (i in k):
                    count+=1
                    #print(i,count)
            row.append(count)
        row.append(sum(row))
        mat.append(row)
    mat.append(list(map(sum,zip(*mat))))
    X = list(range(len(clust)))
    X = list(map(str, X))
    X.append("Total")


    import seaborn as sns
    import matplotlib.pyplot as plt
    fig , ax = plt.subplots(1, 1, figsize=(20, 20))
    from matplotlib.text import Text
    mask = np.zeros((6, 15))
    mask[:, 14] = True
    mask[5:] = True
    sns.heatmap(mat, cmap='Blues',  mask=mask, ax = ax)
    sns.heatmap(mat, alpha=0, cbar=False, annot=True, cmap='Blues', fmt='.4g', annot_kws={"size": 20, "color": "g"}, ax = ax)


    #ig, ax = plt.subplots(1, 1, figsize=(4, 6))
    #sns.heatmap(mat, annot=True, cmap='Blues', fmt='.4g')

    # find your QuadMesh object and get array of colors
    #quadmesh = ax.findobj(QuadMesh)[0]
    #facecolors = quadmesh.get_facecolors()

    # make colors of the last column white
    #facecolors[np.arange(14, 105, 15)] = np.array([1, 1, 1, 1,1,1])

    # set modified colors
   # quadmesh.set_facecolors = facecolors

    # set color of all text to black
    #for i in ax.findobj(Text):
     #   i.set_color('black')

    # move x ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')


    # ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    #ax.set_xlabel('\nPredicted Values')
    #ax.set_ylabel('Actual Values ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(genre)
    ax.yaxis.set_ticklabels(X)

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    print(mat)






