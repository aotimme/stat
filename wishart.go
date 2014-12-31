package stat

import (
  "math"
  "math/rand"
  "fmt"

  "github.com/skelterjohn/go.matrix"
)


type Wishart struct {
  norm Normal
  nu int
}

func NewWishart(sigma *matrix.DenseMatrix, nu int) (wish Wishart) {
  p := sigma.Rows()
  wish.norm = NewNormal(make([]float64, p), sigma)
  wish.nu = nu
  return
}

func (wish *Wishart) Sample(r *rand.Rand) (sigma *matrix.DenseMatrix) {
  X := wish.norm.SampleMultiple(wish.nu, r)
  XT := X.Transpose()
  sigma, err := XT.TimesDense(X)
  if err != nil {
    panic(err)
  }
  p := sigma.Rows()
  for i := 0; i < p; i++ {
    for j := 0; j < p; j++ {
      if sigma.Get(i, j) != sigma.Get(j, i) {
        fmt.Printf("sigma (i,j) != (j,i)\n")
      }
    }
  }
  return
}

func (wish *Wishart) LogDensity(X *matrix.DenseMatrix) (d float64) {
  n, p := wish.nu, len(wish.norm.mean)
  nf, pf := float64(n), float64(p)

  // numerator:
  // |X|^{(n-p-1)/2}
  d = (nf - pf - 1.0) * LogDet(X) / 2.0
  covInv := wish.norm.getInverseCovariance()
  // exp(-V^{-1} X / 2)
  tmp, err := covInv.TimesDense(X)
  if err != nil {
    // could also return -Inf...
    panic(err)
  }
  for i := 0; i < tmp.Rows(); i++ {
    d += tmp.Get(i, i) / 2.0
  }

  // denominator:
  // 2^{np/2}
  d -= nf * pf * math.Log(2) / 2.0
  // |V|^{n/2}
  d -= nf * wish.norm.getLogDetCov() / 2.0
  // \Gamma_p(n/2)
  d -= LogMvGamma(nf / 2.0, int64(p))

  return
}

func (wish *Wishart) Density(X *matrix.DenseMatrix) float64 {
  return math.Exp(wish.LogDensity(X))
}
