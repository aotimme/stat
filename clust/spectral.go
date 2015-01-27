package clust

import (
  "math"
  "sort"

  "github.com/skelterjohn/go.matrix"
)


// see: http://www.informatik.uni-hamburg.de/ML/contents/people/luxburg/publications/Luxburg07_tutorial.pdf
type Spectral struct {
  k int
  // TODO: type (normalized, unnormalized, symmetric, random walk...)
}

func NewSpectral(k int) (self Spectral) {
  self.k = k
  return
}

// for sorting...
type eigen struct {
  val float64
  vector []float64
}
type byVal []eigen
func (v byVal) Len() int           { return len(v) }
func (v byVal) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v byVal) Less(i, j int) bool { return v[i].val < v[j].val }

// Currently implements normalized spectral clustering according to
// Ng, Jordan, and Weiss (2002).
// Note that rather than taking in the raw data points, this function
// takes in a similarity matrix W (which could also be a matrix
// representing a graph).
func (self *Spectral) Cluster(W [][]float64) (assign []int) {
  // compute the (normalized) Laplacian
  n := len(W)
  d := make([]float64, n)
  for i, w := range W {
    for _, wj := range w {
      d[i] += wj
    }
    d[i] = 1.0 / math.Sqrt(d[i])
  }
  L := matrix.Zeros(n, n)
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      elem := 0.0
      if i == j {
        elem = 1.0
      }
      elem -= d[i] * W[i][j] * d[j]
      L.Set(i, j, elem)
    }
  }
  // compute the first k eigenvectors of L
  V, D, err := L.Eigen()
  if err != nil {
    panic(err)
  }
  eigs := make([]eigen, n)
  for i := 0; i < n; i++{
    vec := make([]float64, n)
    V.BufferCol(i, vec)
    eigs[i] = eigen{D.Get(i, i), vec}
  }
  sort.Sort(byVal(eigs))

  // U is the matrix of the first k eigenvectors,
  // with rows normalized to have norm 1
  U := make([][]float64, self.k)
  for i := 0; i < self.k; i++ {
    U[i] = make([]float64, n)
    rowSumSq := 0.0
    for j := 0; j < n; j++ {
      elem := eigs[j].vector[i]
      U[i][j] = elem
      rowSumSq += elem * elem
    }
    for j := 0; j < n; j++ {
      U[i][j] /= math.Sqrt(rowSumSq)
    }
  }

  // cluster the new matrix U via kmeans (each row is a data point)
  km := NewKMeans(self.k)
  return km.Cluster(U, nil)
}
