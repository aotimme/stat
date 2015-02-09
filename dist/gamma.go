package dist

import (
  "math/rand"
  "math"
)

type Gamma struct {
  alpha float64
  beta float64
}

func NewGamma(alpha, beta float64) (gamma Gamma) {
  gamma.alpha = alpha
  gamma.beta = beta
  return
}

func (gamma *Gamma) Sample(r *rand.Rand) float64 {
  // TODO: rejection sample if alpha < 0.75
  if gamma.alpha < 0.75 {
    exp := NewExponential(gamma.beta)
    return RejectionSample(r, gamma.Density, exp.Density, exp.Sample, 1.0)
  }

  // Tadikamalla ACM '73
  // From https://code.google.com/p/gostat/source/browse/stat/gamma.go
  a := gamma.alpha - 1
  b := 0.5 + 0.5 * math.Sqrt(4 * gamma.alpha - 3)
  c := a * (1 + b) / b
  d := (b - 1) / (a * b)
  s := a / b
  p := 1.0 / (2 - math.Exp(-s))
  var x, y float64
  for i := 1; ; i++ {
    u := uniform(r)

    if u > p {
      var e float64
      for e = -math.Log((1 - u) / (1 - p)); e > s; e = e - a/b {
      }
      x = a - b*e
      y = a - x
    } else {
      x = a - b*math.Log(u/p)
      y = x - a
    }
    u2 := uniform(r)
    if math.Log(u2) <= a * math.Log(d * x) - x + y / b + c {
      break
    }
  }
  return x / gamma.beta
}

func (g *Gamma) LogDensity(x float64) float64 {
  return g.alpha * math.Log(g.beta) - lgamma(g.alpha) + (g.alpha - 1) * math.Log(x) - g.beta * x
}

func (g *Gamma) Density(x float64) float64 {
  return math.Exp(g.LogDensity(x))
}
