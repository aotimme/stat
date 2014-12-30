package stat

import (
  "math"
  "math/rand"
  "fmt"

  "github.com/skelterjohn/go.matrix"
)

type Normal struct {
  mean []float64
  cov *matrix.DenseMatrix
  invCov *matrix.DenseMatrix
  chol *matrix.DenseMatrix
  cholT *matrix.DenseMatrix
  logDetCov float64
}

func (norm *Normal) getLogDetCov() float64 {
  if norm.logDetCov == 0.0 {
    norm.logDetCov = LogDet(norm.cov)
  }
  return norm.logDetCov
}

func (norm *Normal) getChol() *matrix.DenseMatrix {
  if norm.chol == nil {
    chol, err := norm.cov.Cholesky()
    if err != nil {
      fmt.Printf("cov = %v\n", norm.cov)
      fmt.Printf("NOT SPD (det = %v)\n", norm.cov.Det())
      panic(err)
    }
    norm.chol = chol
  }
  return norm.chol
}

func (n *Normal) getInverseCovariance() *matrix.DenseMatrix {
  if n.invCov == nil {
    invCov, err := n.cov.Inverse()
    if err != nil {
      panic(err)
    }
    n.invCov = invCov
  }
  return n.invCov
}

func (norm *Normal) getCholT() *matrix.DenseMatrix {
  if norm.cholT == nil {
    chol := norm.getChol()
    cholT := chol.Copy()
    err := cholT.TransposeInPlace()
    if err != nil {
      panic(err)
    }
    norm.cholT = cholT
  }
  return norm.cholT
}

func (norm *Normal) CacheComputations() {
  _ = norm.getCholT()
  _ = norm.getLogDetCov()
  return
}

func NewNormal(mean []float64, cov *matrix.DenseMatrix) (norm Normal) {
  norm.mean = mean
  norm.cov = cov.Copy()
  norm.invCov, norm.chol, norm.cholT = nil, nil, nil
  return
}

func (norm *Normal) Sample(r *rand.Rand) []float64 {
  s := norm.SampleMultiple(1, r)
  return s.RowCopy(0)
}

func (norm *Normal) SampleMultiple(n int, r *rand.Rand) (s *matrix.DenseMatrix) {
  p := len(norm.mean)
  ss := make([]float64, n * p)
  for i := 0; i < len(ss); i++ {
    if r != nil {
      ss[i] = r.NormFloat64()
    } else {
      ss[i] = rand.NormFloat64()
    }
  }
  X := matrix.MakeDenseMatrix(ss, n, p)
  cholT := norm.getCholT()
  s, err := X.TimesDense(cholT)
  if err != nil {
    panic(err)
  }
  for i := 0; i < s.Rows(); i++ {
    for j := 0; j < s.Cols(); j++ {
      s.Set(i, j, s.Get(i,j) + norm.mean[j])
    }
  }
  return
}

func (n *Normal) LogPdf(x []float64) float64 {
  p := len(n.mean)
  covInv := n.getInverseCovariance()
  quad := 0.0
  for i := 0; i < p; i++ {
    for j := 0; j < p; j++ {
      quad += (x[i] - n.mean[i]) * covInv.Get(i, j) * (x[j] - n.mean[j])
    }
  }
  return -0.5 * (float64(p) * math.Log(2 * math.Pi) + n.getLogDetCov() + quad)
}
