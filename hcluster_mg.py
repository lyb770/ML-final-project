import clusters_mg
import sys

def cluster(docs, words, data):

    clust = clusters_mg.hcluster(data, distance=clusters_mg.cosine)
    print('clusters by cosine coefficient')
    clusters_mg.printclust(clust, labels=docs)
    clusters_mg.drawdendrogram(clust, docs, jpeg='docsclustcosine.jpg')

    clust = clusters_mg.hcluster(data, distance=clusters_mg.manhattan)
    print('clusters by manhattan distance')
    clusters_mg.printclust(clust, labels=docs)
    clusters_mg.drawdendrogram(clust, docs, jpeg='docsclustmanhattan.jpg')

    clust=clusters_mg.hcluster(data,distance=clusters_mg.pearson)
    print ('clusters by pearson correlation')
    clusters_mg.printclust(clust,labels=docs)
    clusters_mg.drawdendrogram(clust,docs,jpeg='docsclustpearson.jpg')

    clust=clusters_mg.hcluster(data,distance=clusters_mg.tanimoto)
    print ('clusters by tanimoto coefficient')
    clusters_mg.printclust(clust,labels=docs)
    clusters_mg.drawdendrogram(clust,docs,jpeg='docsclusttanimoto.jpg')

    clust=clusters_mg.hcluster(data,distance=clusters_mg.euclidean)
    print ('clusters by euclidean distance')
    clusters_mg.printclust(clust,labels=docs)
    clusters_mg.drawdendrogram(clust,docs,jpeg='docsclusteuclidean.jpg')