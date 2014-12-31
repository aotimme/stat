package stat

import (
  "math/rand"
  "math"
)

type Beta struct {
  // could just use Dirichlet
  a, b float64
}

func NewBeta(a, b float64) (beta Beta) {
  beta.a = a
  beta.b = b
  return
}

func (beta *Beta) Sample(r *rand.Rand) float64 {
  gammaA, gammaB := NewGamma(beta.a, 1),  NewGamma(beta.b, 1)
  a, b := gammaA.Sample(r), gammaB.Sample(r)
  return a / (a + b)
}

func (beta *Beta) LogDensity(x float64) float64 {
  if x < 0 || x > 1 {
    return math.Inf(-1)
  }
  return (beta.a - 1.0) * math.Log(x) + (beta.b - 1.0) * math.Log(1.0 - x)  + LogGamma(beta.a + beta.b) - LogGamma(beta.a) - LogGamma(beta.b)
}

func (beta *Beta) Density(x float64) float64 {
  return math.Exp(beta.LogDensity(x))
}
