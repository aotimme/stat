package main

import (
  "fmt"
  "math/rand"

  "github.com/skelterjohn/go.matrix"
  "github.com/aotimme/stat/dist"
  "github.com/aotimme/stat/clust"
)

func main() {
  norms := make([]dist.MVNormal, 2)
  cov0 := matrix.Eye(2)
  norms[0] = dist.NewMVNormal([]float64{1.0, 1.0}, cov0)
  cov1 := matrix.Eye(2)
  norms[1] = dist.NewMVNormal([]float64{-1.0, -1.0}, cov1)
  X := make([][]float64, 500)
  r := rand.New(rand.NewSource(20))
  for i := 0; i < 250; i++ {
    X[i] = norms[0].Sample(r)
    X[i+250] = norms[1].Sample(r)
  }

  km := clust.NewKMeans(2)
  _ = km.Cluster(X, r)
  fmt.Printf("centroids: %v\n", km.Centroids())
}
