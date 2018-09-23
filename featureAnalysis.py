##Imports##
import numpy as np
import networkx as nx

class featureSelection():
    #Various functions for producing features from a graph

    def __init__(self, graph_list):
        self.graph_list = graph_list #input list of graphs as a networkx object
        self.numNodes = len(graph_list[0])
    def eigenvectorCentrality(self, numFeatures = 15):
        #computes eigen vector centrality for the graph and returns a np matrix containing
        #eigenvector centrality values for each graph
        #numFeatures selects the number of values to return for each graph, returns values for the
        #first numFeatures nodes
        #default numFeatures is 15
        featureMatrix = []
        for graph in self.graph_list:
            centrality = nx.eigenvector_centrality(graph)
            eigenVectorDict = dict(sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items()))
            list_partial = list(eigenVectorDict.values())[0:numFeatures]
            if len(list_partial) == numFeatures:
                featureMatrix.append(list_partial)
        return np.stack(featureMatrix)

