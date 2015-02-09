package dist

import (
  "math"
  "math/rand"
)

type LogNormal struct {
  mu, sigmasq float64
  norm *Normal
}

func NewLogNormal(mu, sigmasq float64) (ln LogNormal) {
  ln.mu = mu
  ln.sigmasq = sigmasq
  n := NewNormal(mu, sigmasq)
  ln.norm = &n
  return
}

func (ln *LogNormal) Sample(r *rand.Rand) float64 {
  return math.Exp(ln.norm.Sample(r))
}

func (ln *LogNormal) LogDensity(x float64) float64 {
  diff := math.Log(x) - ln.mu
  exp := - diff * diff / (2 * ln.sigmasq)
  pre := -math.Log(x) - 0.5 * math.Log(ln.sigmasq) - 0.5 * math.Log(2 * math.Pi)
  return pre + exp
}

func (ln *LogNormal) Density(x float64) float64 {
  return math.Exp(ln.LogDensity(x))
}
