import networkx as nx
import numpy as np
from sklearn.cluster import AffinityPropagation
#Affinity propogation methods

class affProp():
    #input is a networkx graph object
    def __init__(self, inputGraph, inputGraphList):
        self.G = nx.to_numpy_matrix(inputGraph)
    
    def clusterOnceSingle(self, damping):
        af = AffinityPropagation(np.diagonal(G)).fit(G)
        self.cluster_centers_indices = af.cluster_centers_indices_
        self.labels = af.labels_
        self.n_clusters_ = len(cluster_centers_indices)
        return self.n_clusters



