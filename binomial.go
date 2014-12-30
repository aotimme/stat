package stat

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

func (bin *Binomial) Pdf(x int64) float64 {
  n := bin.n
  p := bin.bern.p
  return float64(Choose(n, x)) * math.Pow(p, float64(x)) * math.Pow(1.0 - p, float64(n - x))
}

func (bin *Binomial) LogPdf(x int64) float64 {
  return float64(x) * math.Log(bin.bern.p) + float64(bin.n - x) * math.Log(1.0 - bin.bern.p) + Lchoose(bin.n, x)
}
