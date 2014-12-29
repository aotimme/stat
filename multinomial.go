package stat

import (
  "math/rand"
)

type Multinomial struct {
  cat Categorical
}

func NewMultinomial(p []float64) (mult Multinomial) {
  mult.cat = NewCategorical(p)
  return
}

func (mult *Multinomial) Sample(n int, r *rand.Rand) (s []float64) {
  s = make([]float64, len(mult.cat.p))
  for i := 0; i < n; i++ {
    idx := mult.cat.Sample(r)
    s[idx]++
  }
  return
}
