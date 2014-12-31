package dist

import (
  "math/rand"
  "math"
)

type Dirichlet struct {
  alpha []float64

  sumAlpha float64
  logNormalizer float64
}

func NewDirichlet(alpha []float64) (dir Dirichlet) {
  dir.alpha = alpha
  dir.sumAlpha, dir.logNormalizer = 0.0, 0.0
  return
}

func (dir *Dirichlet) Sample(r *rand.Rand) []float64 {
  s := make([]float64, len(dir.alpha))
  sum := 0.0
  for i, a := range dir.alpha {
    g := NewGamma(a, 1)
    s[i] = g.Sample(r)
    sum += s[i]
  }
  for i := 0; i < len(s); i++ {
    s[i] /= sum
  }
  return s
}

func (dir *Dirichlet) getSumAlpha() float64 {
  if dir.sumAlpha == 0.0 {
    for _, a := range dir.alpha {
      dir.sumAlpha += a
    }
  }
  return dir.sumAlpha
}

func (dir *Dirichlet) getLogNormalizer() float64 {
  if dir.logNormalizer == 0.0 {
    for _, a := range dir.alpha {
      dir.logNormalizer += LogGamma(a)
    }
    dir.logNormalizer -= LogGamma(dir.getSumAlpha())
  }
  return dir.logNormalizer
}

func (dir *Dirichlet) LogDensity(x []float64) (logpdf float64) {
  for i, xx := range x {
    logpdf += (dir.alpha[i] - 1.0) * math.Log(xx)
  }
  logpdf -= dir.logNormalizer
  return
}

func (dir *Dirichlet) Density(x []float64) float64 {
  return math.Exp(dir.LogDensity(x))
}
