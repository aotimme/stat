package dist

import (
  "math"
  "math/rand"
)

type ChiSquared struct {
  dof int64
}

func NewChiSquared(dof int64) (x ChiSquared) {
  x.dof = dof
  return
}

func (x *ChiSquared) Sample(r *rand.Rand) (s float64) {
  for i := int64(0); i < x.dof; i++ {
    z := stdnormal(r)
    s += z * z
  }
  return
}

func (x *ChiSquared) LogDensity(v float64) float64 {
  doff := float64(x.dof)
  norm := doff * math.Log(2.0) / 2 + lgamma(doff / 2)
  return (doff / 2 - 1) * math.Log(v) - v / 2 - norm
}

func (x *ChiSquared) Density(v float64) float64 {
  return math.Exp(x.LogDensity(v))
}
