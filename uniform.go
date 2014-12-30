package stat

import (
  "math"
  "math/rand"
)

type Uniform struct {
  a, b float64
}

func NewUniform(a, b float64) (unif Uniform) {
  unif.a = a
  unif.b = b
  return
}

func (unif *Uniform) Sample(r *rand.Rand) float64 {
  u := NextUniform(r)
  return u * (unif.b - unif.a) + unif.a
}

func (unif *Uniform) Pdf(x float64) float64 {
  if x < unif.a || x > unif.b {
    return 0.0
  }
  return 1.0 / (unif.b - unif.a)
}

func (unif *Uniform) LogPdf(x float64) float64 {
  if x < unif.a || x > unif.b {
    return math.Inf(-1)
  }
  return -math.Log(unif.b - unif.a)
}
