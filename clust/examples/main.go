package main

import (
  "fmt"
  "math/rand"

  "github.com/skelterjohn/go.matrix"
  "github.com/aotimme/stat/dist"
  "github.com/aotimme/stat/clust"
)

func generate(r *rand.Rand) (X [][]float64) {
  norms := make([]dist.MVNormal, 2)
  cov0 := matrix.Eye(2)
  norms[0] = dist.NewMVNormal([]float64{1.0, 1.0}, cov0)
  cov1 := matrix.Eye(2)
  norms[1] = dist.NewMVNormal([]float64{-1.0, -1.0}, cov1)
  X = make([][]float64, 500)
  for i := 0; i < 250; i++ {
    X[i] = norms[0].Sample(r)
    X[i+250] = norms[1].Sample(r)
  }
  return
}

func runKMeans() {
  r := rand.New(rand.NewSource(20))
  X := generate(r)
  km := clust.NewKMeans(2)
  _ = km.Cluster(X, r)
  fmt.Printf("centroids: %v\n", km.Centroids())
}

func runGMM() {
  r := rand.New(rand.NewSource(20))
  X := generate(r)
  fmt.Printf("-------------\nMLE:\n")
  gmm := clust.NewGMM(2)
  gmm.ClusterMLE(X)
  for i := 0; i < 2; i++ {
    fmt.Printf("Cluster %d:\n", i + 1)
    fmt.Printf("  mu: %v\n", gmm.Mean(i))
    fmt.Printf("  sigma: %v\n", gmm.Covariance(i))
    fmt.Printf("  pi: %v\n", gmm.Proportion(i))
  }
  fmt.Printf("-------------\nMAP:\n")
  gmm = clust.NewGMM(2)
  gmm.ClusterMAP(X)
  for i := 0; i < 2; i++ {
    fmt.Printf("Cluster %d:\n", i + 1)
    fmt.Printf("  mu: %v\n", gmm.Mean(i))
    fmt.Printf("  sigma: %v\n", gmm.Covariance(i))
    fmt.Printf("  pi: %v\n", gmm.Proportion(i))
  }
}

func main() {
  runKMeans()
  runGMM()
}
