package dist

import (
  "math"
  "math/rand"
)

type Multinomial struct {
  n int64
  cat Categorical
}

func NewMultinomial(n int64, p []float64) (mult Multinomial) {
  mult.n = n
  mult.cat = NewCategorical(p)
  return
}

func NewMultinomialUnnormalized(n int64, p []float64) (mult Multinomial) {
  mult.n = n
  mult.cat = NewCategoricalUnnormalized(p)
  return
}

func (mult *Multinomial) Sample(r *rand.Rand) (s []int64) {
  s = make([]int64, len(mult.cat.p))
  for i := int64(0); i < mult.n; i++ {
    idx := mult.cat.Sample(r)
    s[idx]++
  }
  return
}

func (mult *Multinomial) Density(x []int64) (d float64) {
  return math.Exp(mult.LogDensity(x))
}

func (mult *Multinomial) LogDensity(x []int64) (d float64) {
  sum := int64(0)
  for _, xx := range x {
    sum += xx
  }
  d = lgamma(float64(sum) + 1.0)
  for _, xx := range x {
    d -= lgamma(float64(xx) + 1.0)
  }
  for i, xx := range x {
    d += float64(xx) * math.Log(mult.cat.p[i])
  }
  return
}
