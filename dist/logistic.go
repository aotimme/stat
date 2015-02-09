package dist

import (
  "math"
  "math/rand"
)

type Logistic struct {
  mu, s float64
}

func NewLogistic(mu, s float64) (log Logistic) {
  log.mu, log.s = mu, s
  return
}

func (log *Logistic) Sample(r *rand.Rand) float64 {
  return logit(uniform(r))
}

func (log *Logistic) LogDensity(x float64) float64 {
  num := - (x - log.mu) / log.s
  den := math.Log(log.s) + 2.0 * math.Log(1.0 + math.Exp(num))
  return num - den
}
