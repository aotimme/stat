package dist

import (
  "math"
  "math/rand"
)

type Polya struct {
  n int64
  dir *Dirichlet
}

func NewPolya(n int64, alpha []float64) (p Polya) {
  p.n = n
  dir := NewDirichlet(alpha)
  p.dir = &dir
  return
}

func (p *Polya) Sample(r *rand.Rand) []int64 {
  ps := p.dir.Sample(r)
  mult := NewMultinomial(p.n, ps)
  return mult.Sample(r)
}

func (p *Polya) LogDensity(x []int64) float64 {
  sumAlpha := p.dir.getSumAlpha()
  sumN := 0.0
  for _, xx := range x {
    sumN += float64(xx)
  }
  ld := lgamma(sumAlpha) - lgamma(sumN + sumAlpha)
  for k, xx := range x {
    ld += lgamma(float64(xx) + p.dir.alpha[k]) - lgamma(p.dir.alpha[k])
  }
  return ld
}

func (p *Polya) Density(x []int64) float64 {
  return math.Exp(p.LogDensity(x))
}
