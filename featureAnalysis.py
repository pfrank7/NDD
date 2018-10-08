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
        #numFeatures selects the number of values to return for each graph, returns values for the features with highest variance
        featureMatrix = []
        for graph in self.graph_list:
            centrality = nx.eigenvector_centrality(graph)
            eigenVectorDict = dict(sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items()))
            list_ = list(eigenVectorDict.values())
            if len(list_) == self.numNodes:
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
            
        

