##Imports##
import numpy as np
import networkx as nx
from networkx import normalized_laplacian_matrix
import sys

class featureSelection():

    
    '''Initialization creates a list of all of the graphs removing any with
    less than the correct number of vertices '''
    
    def __init__(self, graph_list):
        X = []
        len_ = [len(row) for row in graph_list]
        ind = np.argmax(np.bincount(len_))
        if ind >= len(graph_list):
            ind = 0
        cutoff = len_[ind]
        for row in graph_list:
            if len(row) == cutoff:
                X.append(row)
        self.graph_list = X
        self.all = len(X)
    '''The following flattens the adjacency matrix of each graph.
    It takes only the top diagnal of values and returns a feature 
    matrix with those values as features. '''
    
    def adjacencyMatrixFeatures(self):
        featureMat = []
        for G in self.graph_list:
            adjMat = nx.to_numpy_matrix(G)
            topHalf = np.triu_indices(len(adjMat))
            row = adjMat[topHalf]
            featureMat.append(row)
        featureMat = np.vstack(featureMat)
        return featureMat
    
    '''The following calculated the eigenvector centrality
    of each vertex and returns the largest average value
    features for the number of features selected'''
    
    def eigenvectorCentrality(self, numFeatures = 0):
        #computes eigen vector centrality for the graph and returns a np matrix containing
        #eigenvector centrality values for each graph
        if numFeatures == 0:
                numFeatures = self.all
        featureMatrix = []
        for graph in self.graph_list:
            centrality = nx.eigenvector_centrality(graph)
            eigenVectorDict = dict(sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items()))
            list_ = list(eigenVectorDict.values())
            featureMatrix.append(list_)
        featureMatrix = np.stack(featureMatrix).astype("float")
        # Select highest feature centrality values
        averages = np.mean(featureMatrix, axis=0)
        featInds = np.argsort(averages)[-numFeatures:]
        return featureMatrix[:,featInds]
    
    '''A helper function to calculate the variance of each feature'''
    
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
    

    ''' A helper function to calculate the eigenvalues of
    each graph laplacian '''
    
    def _calculate_eigenvalues(self):
        # Return eigenvalues of the Laplacian
        eigvals = []
        for G in self.graph_list:
            L = normalized_laplacian_matrix(G)
            e = list(np.linalg.eigvals(L.A))
            eigvals.append(e)
        return eigvals

    '''The following returns the eigenvalues of each
    graph as its features. It returns the highest
    variance features'''
    
    def calc_eigval_feature_matrix(self, numFeatures = 0):
        if numFeatures == 0:
            numFeatures = self.all
        X = self._calculate_eigenvalues()
        X = np.matrix(X)
        varValues = self.getVariance(X)
        featInds = np.argsort(varValues)[-numFeatures:]
        return X[:,featInds]

    
    '''The following functions create a feature matrix containing
    the one and two khop locality values for each matrix.
    The functions with 1 are for networkx version 1.
    Each selects the highest variance number of features'''

        
    def khop_locality_1(self, G):
        embed = []

        for node in G.nodes_iter():

            one_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
            two_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())

            embed += len(G.subgraph(one_hop).edges()), len(G.subgraph(two_hop).edges())

        return embed

    def getKhopFeatMat1(self, numFeatures = 0):
        if numFeatures == 0:
            numFeatures = self.all
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
    
    def khop_locality(self, G):
        embed = []
        for node in G.nodes:
            one_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
            two_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
            embed += len(G.subgraph(one_hop).edges()), len(G.subgraph(two_hop).edges())
        return embed

    def getKhopFeatMat(self, numFeatures = 0):
        if numFeatures == 0:
            numFeatures = self.all
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
    
    ''' Remove features that are all zero '''
    def removeZeroColumns(self, mat):
        all_zeros = []
        for i in range(mat.shape[1]):
            if sum(mat[:,i]) == 0:
                all_zeros.append(i)
        mat = np.delete(mat, all_zeros, 1)
        return mat
        

