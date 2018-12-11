'''File containing all clustering algorithms.
    pfrank7@jhu.edu
'''
import numpy as np
import math
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
'''The following algorithms take in a similarity or distance matrix
   Each chooses the best output, varying parameters.
   Affinity propagation takes in a similarity matrix and varies the preference value
   Agglomerative clustering takes in a distance matrix and varies the number of clusters
   DBSCAN takes in a distance matrix and varies the epsilon value.
   Silhouette coefficient is used to select the best parameter setting'''

def AffinityProp_BestSil(inputMat, damp = .9, preferenceInit = 0.0, preferenceInc = .002):
    # Identify the maximum parameter values while clustering #
    maxPre = 0
    maxSil = -1
    maxElement = inputMat.max()
    pre = preferenceInit
    trials = []
    while pre < maxElement:
        # vary the preference value #
        pre += preferenceInc
        af = AffinityPropagation(damping = damp, preference = pre, affinity="precomputed").fit(inputMat)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters = len(cluster_centers_indices)
        # Determine if the parameters maximize the silhouette coefficient #
        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ss = silhouette_score(inputMat, labels)
            if (ss > maxSil):
                maxDamp = damp
                maxPre = pre
                maxSil = ss
        trials.append([n_clusters, ss, pre])
    return maxPre, maxSil, trials

def AgglomerativeClustering_BestSil(inputMat, clustInit = 2, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSil = -1
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        ag = AgglomerativeClustering(n_clusters=c, affinity="precomputed", linkage = 'average').fit(inputMat)
        labels = ag.labels_
        n_clusters = c

        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ss = silhouette_score(inputMat, labels)
            if (ss > maxSil):
                maxClusters = c
                maxSil = ss
        trials.append([c, ss])
    return maxClusters, maxSil, trials

def DBSCANClustering_BestSil(inputMat, epsInit = 0, epsInc = 0.002):
    # Identify the maximum parameter values while clustering #
    maxEps = 0.0
    maxSil = -1
    maxElement = inputMat.max()
    e = epsInit
    trials = []
    while e < maxElement:
        # vary the eps value #
        e += epsInc
        db = DBSCAN(metric="precomputed").fit(inputMat)
        labels = db.labels_
        n_clusters = len(set(labels))
        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ss = silhouette_score(inputMat, labels)
            if (ss > maxSil):
                maxSil = ss
                maxEps = e
            trials.append([n_clusters, ss, e])
    return maxEps, maxSil, trials

'''The following algorithms take in a feature matrix.
   Each chooses the best output, varying parameters.
   MiniBatch K-Means takes in a feature matrix and varies the number of clusters
   Batch size is set to 1/3 of the total number of values or 100 whichever is smaller.
   K means takes in a feature matrix and varies the number of clusters
   GMM takes in a feature matrix and varies the number of clusters.
   '''

def MiniBatch_BestSil(inputMat, clustInit = 1, clustInc = 1, batchSize = 100, clustMax = 50):
    if batchSize > len(inputMat):
        batchSize = math.ceil(len(inputMat)/3)
    maxClusters = 0
    maxSil = 0
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        mb = MiniBatchKMeans(n_clusters = c, batch_size = batchSize).fit(inputMat)
        cluster_centers_ = mb.cluster_centers_
        labels = mb.labels_

        # capture best cluster number #
        ss = silhouette_score(inputMat, labels)
        if (ss > maxSil):
            maxClusters = c
            maxSil = ss
        trials.append([c, ss])
    return maxClusters, maxSil, trials

def GMM_BestSil(inputMat, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxSil = 0
    trials = []
    c = clustInit
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        gm = GaussianMixture(n_components = c)
        gm.fit(inputMat)
        labels = gm.predict(inputMat)

        # capture best cluster number #
        ss = silhouette_score(inputMat, labels)
        if (ss > maxSil):
            maxClusters = c
            maxSil = ss
        trials.append([c, ss])
    return maxClusters, maxSil, trials

def KMeansClustering_BestSil(inputMat, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxSil = 0
    trials = []
    c = clustInit
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        km = KMeans(n_clusters = c).fit(inputMat)
        cluster_centers_ = km.cluster_centers_
        labels = km.labels_

        # capture best cluster number #
        ss = silhouette_score(inputMat, labels)
        if (ss > maxSil):
            maxClusters = c
            maxSil = ss
        trials.append([c, ss])
    return maxClusters, maxSil, trials

def SpectralClust_BestSil(inputMat, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxSil = 0
    trials = []
    c = clustInit
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        sc = SpectralClustering(n_clusters = c).fit(inputMat)
        labels = sc.labels_
        
        # capture best cluster number #
        ss = silhouette_score(inputMat, labels)
        if (ss > maxSil):
            maxClusters = c
            maxSil = ss
        trials.append([c, ss])
    return maxClusters, maxSil, trials

''' Using ARI to choose the best parameters '''

def AffinityProp_BestAri(inputMat, labels_true, damp = .9, preferenceInit = 0.0, preferenceInc = .002):
    # Identify the maximum parameter values while clustering #
    maxPre = 0
    maxAri = -1
    maxElement = inputMat.max()
    pre = preferenceInit
    trials = []
    while pre < maxElement:
        # vary the preference value #
        pre += preferenceInc
        af = AffinityPropagation(damping = damp, preference = pre, affinity="precomputed").fit(inputMat)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters = len(cluster_centers_indices)
        # Determine if the parameters maximize the silhouette coefficient #
        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ari = adjusted_rand_score(labels_true, labels)
            if (ari > maxAri):
                maxAri = ari
                maxDamp = damp
                maxPre = pre
        trials.append([n_clusters, ari, pre])
    return maxPre, maxAri, trials

def AgglomerativeClustering_BestAri(inputMat, labels_true, clustInit = 2, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxAri = -1
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        ag = AgglomerativeClustering(n_clusters=c, affinity="precomputed", linkage = 'average').fit(inputMat)
        labels = ag.labels_
        n_clusters = c

        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ari = adjusted_rand_score(labels_true, labels)
            if (ari > maxAri):
                maxAri = ari
                maxClusters = c
        trials.append([c, ari])
    return maxClusters, maxAri, trials

def DBSCANClustering_BestAri(inputMat, labels_true, epsInit = 0, epsInc = 0.002):
    # Identify the maximum parameter values while clustering #
    maxEps = 0.0
    maxAri = -1
    maxElement = inputMat.max()
    e = epsInit
    trials = []
    while e < maxElement:
        # vary the eps value #
        e += epsInc
        db = DBSCAN(metric="precomputed").fit(inputMat)
        labels = db.labels_
        n_clusters = len(set(labels))
        if n_clusters <= len(inputMat) - 1 and n_clusters > 1:
            ari = adjusted_rand_score(labels_true, labels)
            if (ari > maxAri):
                maxAri = ari
                maxEps = e
            trials.append([n_clusters, ari, e])
    return maxEps, maxAri, trials



def MiniBatch_BestAri(inputMat, labels_true, clustInit = 1, clustInc = 1, batchSize = 100, clustMax = 50):
    if batchSize > len(inputMat):
        batchSize = math.ceil(len(inputMat)/3)
    maxClusters = 0
    maxAri = 0
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        mb = MiniBatchKMeans(n_clusters = c, batch_size = batchSize).fit(inputMat)
        cluster_centers_ = mb.cluster_centers_
        labels = mb.labels_

        # capture best cluster number #
        ari = adjusted_rand_score(labels_true, labels)
        if (ari > maxAri):
            maxAri = ari
            maxClusters = c
        trials.append([c, ari])
    return maxClusters, maxAri, trials

def GMM_BestAri(inputMat, labels_true, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxAri = 0
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        gm = GaussianMixture(n_components = c)
        gm.fit(inputMat)
        labels = gm.predict(inputMat)

        # capture best cluster number #
        ari = adjusted_rand_score(labels_true, labels)
        if (ari > maxAri):
            maxAri = ari
            maxClusters = c
        trials.append([c, ari])
    return maxClusters, maxAri, trials

def KMeansClustering_BestAri(inputMat, labels_true, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxAri = 0
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        km = KMeans(n_clusters = c).fit(inputMat)
        cluster_centers_ = km.cluster_centers_
        labels = km.labels_

        # capture best cluster number #
        ari = adjusted_rand_score(labels_true, labels)
        if (ari > maxAri):
            maxAri = ari
            maxClusters = c
        trials.append([c, ari])
    return maxClusters, maxAri, trials

def SpectralClust_BestAri(inputMat, labels_true, clustInit = 1, clustInc = 1, clustMax = 50):
    maxClusters = 0
    maxSize = 0
    maxAri = 0
    c = clustInit
    trials = []
    while c < clustMax:
        # vary the cluster number #
        c += clustInc
        
        sc = SpectralClustering(n_clusters = c).fit(inputMat)
        labels = sc.labels_
        
        # capture best cluster number #
        ari = adjusted_rand_score(labels_true, labels)
        if (ari > maxAri):
            maxAri = ari
            maxClusters = c
        trials.append([c, ari])
    return maxClusters, maxAri, trials