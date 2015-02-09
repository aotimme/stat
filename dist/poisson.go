package dist

import (
  "math"
  "math/rand"
)

type Poisson struct {
  lambda float64
}

func NewPoisson(lambda float64) (poisson Poisson) {
  poisson.lambda = lambda
  return
}

func (poisson *Poisson) Sample(r *rand.Rand) int64 {
  l := math.Exp(-poisson.lambda)
  p, k := uniform(r), 0
  for p > l {
    k++
    p *= uniform(r)
  }
  return int64(k)
}

func (poisson *Poisson) LogDensity(k int64) float64 {
  kf := float64(k)
  return kf * math.Log(poisson.lambda) - poisson.lambda - lgamma(kf + 1.0)
}

func (poisson *Poisson) Density(k int64) float64 {
  return math.Exp(poisson.LogDensity(k))
}
