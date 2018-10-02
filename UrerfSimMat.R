
# make sure to put the correct path to rfr_us.R
source('~/Documents/U-Rerf/rfr_us.R')

# number of trees for forest
numtrees <- 100
# the 'k' of k nearest neighbors
k <- 10
# set max depth
depth <- 4

#D = num graphs, m = eigenvector centrality values
X<-as.matrix(read.table("~/Documents/NDD/outfileMat.txt",header=FALSE,sep=" "))

# create a urerf structure which includes the similarity matrix
urerfStructure <- urerf(X, numtrees, k)
SM = urerfStructure$similarityMatrix
write.table(SM, file="~/Documents/NDD/outfileSimMat.txt", row.names=FALSE, col.names=FALSE)
