source('~/Documents/U-Rerf/rfr_us.R')
  
  "number of trees for forest"
  numtrees <- 500
  "the 'k' of k nearest neighbors"
  k <- 10
  "set max depth"
  depth <- 4
  
  "D = num graphs, m = iris features"
  X<-as.matrix(read.table("~/Documents/NDD/output/outfileMat_HNU_d.txt",header=FALSE,sep=" "))

  "create a urerf structure which includes the similarity matrix"
  urerfStructure <- urerf(X, numtrees, k)
  SM = urerfStructure$similarityMatrix
  
  write.table(SM, file="~/Documents/NDD/input/outfileSimMat_HNU_d.txt", row.names=FALSE, col.names=FALSE)

