package dist

import (
  "math"
  "math/rand"
)

type Weibull struct {
  lambda, k float64
}

func NewWeibull(lambda, k float64) (w Weibull) {
  w.lambda, w.k = lambda, k
  return
}

func (w *Weibull) Sample(r *rand.Rand) float64 {
  u := uniform(r)
  return w.lambda * math.Pow(-math.Log(u), 1.0 / w.k)
}

func (w *Weibull) LogDensity(x float64) float64 {
  if x < 0 {
    return math.Inf(-1)
  }
  return math.Log(w.k) - math.Log(w.lambda) + (w.k - 1.0) * (math.Log(x) - math.Log(w.lambda)) - math.Pow(x / w.lambda, w.k)
}

func (w *Weibull) Density(x float64) float64 {
  return math.Exp(w.LogDensity(x))
}
