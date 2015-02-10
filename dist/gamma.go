package dist

import (
  "math/rand"
  "math"
)

type Gamma struct {
  alpha float64
  beta float64
}

func NewGamma(alpha, beta float64) (self Gamma) {
  self.alpha = alpha
  self.beta = beta
  return
}

func (self *Gamma) Sample(r *rand.Rand) float64 {
  // See: http://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
  // Marsaglia and Tsang's Method (gsl implementation)
  if self.alpha <= 1.0 {
    tmpGam := NewGamma(self.alpha + 1.0, self.beta)
    return tmpGam.Sample(r) * math.Pow(uniform(r), 1.0 / self.alpha)
  }

  d := self.alpha - 1.0 / 3.0
  c := 1.0 / (3.0 * math.Sqrt(d))
  var x float64
  for {
    z := stdnormal(r)
    u := uniform(r)
    v := 1.0 + c * z
    v = v * v * v
    if z > -1.0 / c && math.Log(u) < 0.5 * z * z + d * (1.0 - v + math.Log(v)) {
      x = d * v
      break
    }
  }
  return x / self.beta
}

func (g *Gamma) LogDensity(x float64) float64 {
  return g.alpha * math.Log(g.beta) - lgamma(g.alpha) + (g.alpha - 1) * math.Log(x) - g.beta * x
}

func (g *Gamma) Density(x float64) float64 {
  return math.Exp(g.LogDensity(x))
}
