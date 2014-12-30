package stat

import (
  "math"
  "math/rand"
)

type Bernoulli struct {
  p float64
}

func NewBernoulli(p float64) (bern Bernoulli) {
  bern.p = p
  return
}

func (bern *Bernoulli) Sample(r *rand.Rand) (z int64) {
  u := NextUniform(r)
  if u < bern.p {
    z = 1
  } else {
    z = 0
  }
  return
}

func (bern *Bernoulli) Pdf(x int64) float64 {
  if x == 1 {
    return bern.p
  } else if x == 0 {
    return 1.0 - bern.p
  } else {
    return 0.0
  }
}

func (bern *Bernoulli) LogPdf(x int64) float64 {
  if x == 1 {
    return math.Log(bern.p)
  } else if x == 0 {
    return math.Log(1.0 - bern.p)
  } else {
    return math.Inf(-1)
  }
}
