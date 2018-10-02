import numpy as np
import networkx as nx
from edge_fetch import edge_terrier
from featureAnalysis import featureSelection
from sklearn.cluster import KMeans
# Initialize edgelist collector object
graph_collector = edge_terrier('/Users/paigefrank/Library/Python/3.6/bin/aws',filepath='hbn/derivatives/graphs/JHU/')
# Make a generator that yields all edgelists in filepath
filelist = graph_collector.convert_edgelist_all()
# Iterate
graphlist = graph_collector.getGraphs(filelist)
feature_selector = featureSelection(graphlist)
print(feature_selector.eigenvectorCentrality(10))
"""kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans4 = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans2.labels_)
print(kmeans3.labels_)
print(kmeans4.labels_)
print(count)"""
