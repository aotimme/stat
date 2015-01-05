package clust

import (
  "fmt"
  "math"
  "math/rand"

  "github.com/aotimme/stat/dist"
)

type KMeans struct {
  k int
  centroids [][]float64
}

func NewKMeans(k int) (self KMeans) {
  self.k = k
  self.centroids = make([][]float64, k)
  return
}

func (self *KMeans) Centroids() [][]float64 {
  return self.centroids
}

func (self *KMeans) SetCentroids(centroids [][]float64) {
  self.centroids = centroids
  self.k = len(centroids)
}

func euclideansq(x []float64, y []float64) (val float64) {
  for i, _ := range x {
    val += (y[i] - x[i]) * (y[i] - x[i])
  }
  return
}

func euclidean(x []float64, y[]float64) float64 {
  return math.Sqrt(euclideansq(x, y))
}

func minidx(x []float64) (idx int) {
  for i := 1; i < len(x); i++ {
    if x[i] < x[idx] {
      idx = i
    }
  }
  return
}

func min(x []float64) float64 {
  return x[minidx(x)]
}

func (self *KMeans) Initialize(X [][]float64, r *rand.Rand) (assign []int) {
  n := len(X)
  p := len(X[0])

  // KMeans++ initialization
  centroidIdx := make([]int, self.k)
  unif := dist.NewUniform(0.0, float64(n))
  centroidIdx[0] = int(unif.Sample(r))
  for j := 1; j < self.k; j++ {
    dists := make([]float64, n)
    for i, x := range X {
      centroidDists := make([]float64, j)
      for l := 0; l < j; l++ {
        centroidDists[l] = euclidean(x, X[centroidIdx[l]])
      }
      dists[i] = min(centroidDists)
    }
    cat := dist.NewCategorical(dists)
    centroidIdx[j] = int(cat.Sample(r))
  }
  for i := 0; i < self.k; i++ {
    self.centroids[i] = make([]float64, p)
    copy(self.centroids[i], X[centroidIdx[i]])
  }

  assign = make([]int, n)

  totalSqDist := 0.0
  for i, x := range X {
    dists := make([]float64, self.k)
    for j, centroid := range self.centroids {
      dists[j] = euclideansq(x, centroid)
    }
    assign[i] = minidx(dists)
    totalSqDist += dists[assign[i]]
  }

  return
}

func (self *KMeans) Cluster(X [][]float64, r *rand.Rand) (assign []int) {
  epsilon := 1e-6
  maxIters := 200

  p := len(X[0])

  assign = self.Initialize(X, r)

  iter := 0
  prevTotalSqDist := 0.0
  for {

    // reset centroids
    for _, centroid := range self.centroids {
      for j, _ := range centroid {
        centroid[j] = 0.0
      }
    }
    nk := make([]int, self.k)
    for i, x := range X {
      c := assign[i]
      nk[c]++
      for j, xj := range x {
        self.centroids[c][j] += xj
      }
    }
    for i := 0; i < self.k; i++ {
      for j := 0; j < p; j++ {
        self.centroids[i][j] /= float64(nk[i])
      }
    }

    // assignments
    totalSqDist := 0.0
    for i, x := range X {
      dists := make([]float64, self.k)
      for j, centroid := range self.centroids {
        dists[j] = euclideansq(x, centroid)
      }
      assign[i] = minidx(dists)
      totalSqDist += dists[assign[i]]
    }

    fmt.Printf("Iteration %d: %f\n", iter + 1, totalSqDist)

    if iter > 0 {
      if prevTotalSqDist - totalSqDist < epsilon {
        break
      }
    }
    prevTotalSqDist = totalSqDist

    iter++
    if iter > maxIters {
      break
    }
  }

  return
}
