package stat

import (
  "math/rand"
  "math"
)

type Beta struct {
  dir Dirichlet
}

func NewBeta(a, b float64) (beta Beta) {
  beta.dir = NewDirichlet([]float64{a, b})
  return
}

func (beta *Beta) Sample(r *rand.Rand) float64 {
  return beta.dir.Sample(r)[0]
}

func (beta *Beta) LogDensity(x float64) float64 {
  if x < 0 || x > 1 {
    return math.Inf(-1)
  }
  xx := []float64{x, 1.0 - x}
  return beta.dir.LogDensity(xx)
}

func (beta *Beta) Density(x float64) float64 {
  return math.Exp(beta.LogDensity(x))
}
