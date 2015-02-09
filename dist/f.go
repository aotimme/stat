package dist

import (
  "math"
  "math/rand"
)

type F struct {
  d1, d2 int64
  chi1, chi2 *ChiSquared
}

func NewF(d1, d2 int64) (f F) {
  f.d1, f.d2 = d1, d2
  chi1 := NewChiSquared(d1)
  chi2 := NewChiSquared(d2)
  f.chi1, f.chi2 = &chi1, &chi2
  return
}

func (f *F) Sample(r *rand.Rand) float64 {
  return (f.chi1.Sample(r) / float64(f.d1)) / (f.chi2.Sample(r) / float64(f.d2))
}

func (f *F) LogDensity(x float64) float64 {
  d1f := float64(f.d1)
  d2f := float64(f.d2)
  num := 0.5 * (d1f * (math.Log(d1f * x)) + d2f * math.Log(d2f) - (d1f + d2f) * math.Log(d1f * x + d2f))
  den := math.Log(x) + lbeta(0.5 * d1f, 0.5 * d2f)
  return num - den
}

func (f *F) Density(x float64) float64 {
  return math.Exp(f.LogDensity(x))
}
