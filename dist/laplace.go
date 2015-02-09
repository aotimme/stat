package dist

import (
  "math"
  "math/rand"
)

type Laplace struct {
  mu, b float64
}

func NewLaplace(mu, b float64) (l Laplace) {
  l.mu = mu
  l.b = b
  return
}

func (lap *Laplace) Sample(r *rand.Rand) float64 {
  u := uniform(r) - 0.5
  s := 1.0
  if u < 0 {
    s = -1.0
  }
  return lap.mu - lap.b * s * math.Log(1.0 - 2 * math.Abs(u))
}

func (lap *Laplace) LogDensity(x float64) float64 {
  return -math.Log(2.0 * lap.b) - math.Abs(x - lap.mu) / lap.b
}

func (lap *Laplace) Density(x float64) float64 {
  return math.Exp(lap.LogDensity(x))
}
