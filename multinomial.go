package stat

import (
  "math/rand"
)

type Multinomial struct {
  n int
  cat Categorical
}

func NewMultinomial(n int, p []float64) (mult Multinomial) {
  mult.n = n
  mult.cat = NewCategorical(p)
  return
}

func (mult *Multinomial) Sample(r *rand.Rand) (s []float64) {
  s = make([]float64, len(mult.cat.p))
  for i := 0; i < mult.n; i++ {
    idx := mult.cat.Sample(r)
    s[idx]++
  }
  return
}
