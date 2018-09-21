import numpy as np
import networkx as nx
from edge_fetch import edge_terrier
from sklearn.cluster import KMeans
# Initialize edgelist collector object
graph_collector = edge_terrier(filepath='hbn/derivatives/graphs/JHU/')
# Make a generator that yields all edgelists in filepath
filelist = graph_collector.convert_edgelist_all()
# Iterate
listOfLists = []
for edgelist in filelist:
    centrality = nx.eigenvector_centrality(edgelist[0])
    dictFull = dict(sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items()))
    list10 = list(dictFull.values())[0:10]
    if len(list10) == 10:
        listOfLists.append(list10)
X = np.stack(listOfLists)
print(X)
kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans4 = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans2.labels_)
print(kmeans3.labels_)
print(kmeans4.labels_)
