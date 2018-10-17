##Imports##
import numpy as np
import networkx as nx
from networkx import normalized_laplacian_matrix
import sys

class featureSelection():
    #Various functions for producing features from a graph

    def __init__(self, graph_list):
        X = []
        len_ = [len(row) for row in graph_list]
        cutoff = len_[np.argmax(np.bincount(len_))]
        for row in graph_list:
            if len(row) == cutoff:
                X.append(row)
        self.graph_list = X
        
    def eigenvectorCentrality(self, numFeatures = 15):
        #computes eigen vector centrality for the graph and returns a np matrix containing
        #eigenvector centrality values for each graph
        #numFeatures selects the number of values to return for each graph, returns values for the features with highest variance
        featureMatrix = []
        for graph in self.graph_list:
            centrality = nx.eigenvector_centrality(graph)
            eigenVectorDict = dict(sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items()))
            list_ = list(eigenVectorDict.values())
            featureMatrix.append(list_)
        featureMatrix = np.stack(featureMatrix).astype("float")
        varValues = self.getVariance(featureMatrix)
        featInds = np.argsort(varValues)[-numFeatures:]
        return featureMatrix[:,featInds]
    
    def getVariance(self, featureMatrix):
        varValues = []
        for i in range(0, featureMatrix.shape[1]):
            values = featureMatrix[:,i]
            meanSample = np.mean(values)
            sumSample = 0
            for j in range(0, featureMatrix.shape[0]):
                sumSample += (featureMatrix[j,i] - meanSample)**2
            varValues.append(sumSample/(featureMatrix.shape[0] - 1))
        return varValues
    


    def _calculate_eigenvalues(self):
        # Return eigenvalues of the Laplacian
        eigvals = []
        for G in self.graph_list:
            L = normalized_laplacian_matrix(G)
            e = list(np.linalg.eigvals(L.A))
            eigvals.append(e)
        return eigvals


    def calc_eigval_feature_matrix(self, numFeatures):
        X = self._calculate_eigenvalues()
        X = np.matrix(X)
        varValues = self.getVariance(X)
        featInds = np.argsort(varValues)[-numFeatures:]
        return X[:,featInds]


    def khop_locality(self, G):
        embed = []
        for node in G.nodes:
            one_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
            two_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
            embed += len(G.subgraph(one_hop).edges()), len(G.subgraph(two_hop).edges())
        return embed
        
    def khop_locality_1(self, G):
        embed = []

        for node in G.nodes_iter():

            one_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
            two_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())

            embed += len(G.subgraph(one_hop).edges()), len(G.subgraph(two_hop).edges())

        return embed

    def getKhopFeatMat1(self, numFeatures):
        M = []
        for i in range(0, len(self.graph_list)):
            if self.graph_list[i] is not None:
                embed = self.khop_locality_1(self.graph_list[i])
            if embed is not None:
                M.append(embed)
        X = np.matrix(M)
        varValues = self.getVariance(X)
        featInds = np.argsort(varValues)[-numFeatures:]
        return X[:,featInds]
    

    def getKhopFeatMat(self, numFeatures):
        M = []
        for i in range(0, len(self.graph_list)):
            if self.graph_list[i] is not None:
                embed = self.khop_locality(self.graph_list[i])
            if embed is not None:
                M.append(embed)

        # Preserve only the necessary features
        X = np.matrix(M)
        varValues = self.getVariance(X)
        featInds = np.argsort(varValues)[-numFeatures:]
        return X[:,featInds]
        

