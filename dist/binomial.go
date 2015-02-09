package dist

import (
  "math"
  "math/rand"
)

type Binomial struct {
  n int64
  bern Bernoulli
}

func NewBinomial(n int64, p float64) (bin Binomial) {
  bin.bern = NewBernoulli(p)
  bin.n = n
  return
}

func (bin *Binomial) Sample(r *rand.Rand) (x int64) {
  for i := int64(0); i < bin.n; i++ {
    x += bin.bern.Sample(r)
  }
  return
}

func (bin *Binomial) LogDensity(x int64) float64 {
  return float64(x) * math.Log(bin.bern.p) + float64(bin.n - x) * math.Log(1.0 - bin.bern.p) + lchoose(bin.n, x)
}

func (bin *Binomial) Density(x int64) float64 {
  return math.Exp(bin.LogDensity(x))
}
