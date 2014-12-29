package stat

import (
  "math/rand"
)

type Bernoulli struct {
  p float64
}

func NewBernoulli(p float64) (bern Bernoulli) {
  bern.p = p
  return
}

func (bern *Bernoulli) Sample(r *rand.Rand) (z int) {
  u := NextUniform(r)
  if u < bern.p {
    z = 1
  } else {
    z = 0
  }
  return
}
