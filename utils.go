package stat

import (
  "math"
  "math/rand"
  "math/big"

  "github.com/skelterjohn/go.matrix"
)

func NextUniform(r *rand.Rand) (u float64) {
  if r != nil {
    u = r.Float64()
  } else {
    u = rand.Float64()
  }
  return
}

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

func RejectionSample(r *rand.Rand, targetLogPdf func(float64) float64, sourceLogPdf func(float64) float64, source func(*rand.Rand) float64, K float64) float64 {
  x := source(r)
  for math.Log(NextUniform(r)) >= targetLogPdf(x) - sourceLogPdf(x) - math.Log(K) {
    x = source(r)
  }
  return x
}

// uses `math.Lgamma`, but returns just a float, without the sign
func LogGamma(x float64) float64 {
  lgamma, sign := math.Lgamma(x)
  if sign == 1 {
    return lgamma
  } else {
    return -lgamma
  }
}

func Choose(n, k int64) int64 {
  bigIntPtr := &big.Int{}
  bigIntPtr.Binomial(n, k)
  return bigIntPtr.Int64()
}

func Lchoose(n, k int64) float64 {
  bigIntPtr := &big.Int{}
  bigIntPtr.Binomial(n, k)
  z := bigIntPtr.Int64()
  return math.Log(float64(z))
}
