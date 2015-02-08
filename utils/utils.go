package utils

import (
  "math"
  "math/big"

  "github.com/skelterjohn/go.matrix"
)

func LogDet(m *matrix.DenseMatrix) (logdet float64) {
  // NOTE: stabler version of `math.Log(m.Det())` (which unfortunately is faster)
  //logdet = math.Log(m.Det())
  _, D, err := m.Eigen()
  if err != nil {
    panic(err)
  }
  for i := 0; i < D.Rows(); i++ {
    logdet += math.Log(D.Get(i,i))
  }
  return
}

// LogGamma uses `math.Lgamma`, but returns just a float, without the sign
// (assumes the variable passed in is positive)
func LogGamma(x float64) float64 {
  lgamma, _ := math.Lgamma(x)
  return lgamma
  //if sign == 1 {
  //  return lgamma
  //} else {
  //  return -lgamma
  //}
}

// http://en.wikipedia.org/wiki/Multivariate_gamma_function
func LogMvGamma(x float64, p int64) (lmvgamma float64) {
  pf := float64(p)
  lmvgamma = (pf * (pf - 1.0)) / 4.0 * math.Log(math.Pi)
  for i := int64(0); i < p; i++ {
    lmvgamma += LogGamma(x + (1.0 - float64(i)/2.0))
  }
  return
}

func Choose(n, k int64) int64 {
  bigIntPtr := &big.Int{}
  bigIntPtr.Binomial(n, k)
  return bigIntPtr.Int64()
}

func Lchoose(n, k int64) float64 {
  nn := float64(n)
  kk := float64(k)
  nk := float64(n + k)
  return LogGamma(nk) - LogGamma(nn) - LogGamma(kk)
}
