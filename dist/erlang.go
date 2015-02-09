package dist

import (
  "math/rand"
  "math"
)

type Erlang struct {
  k int64
  lambda float64
}

func NewErlang(k int64, lambda float64) (e Erlang) {
  e.k, e.lambda = k, lambda
  return
}

func (e *Erlang) Sample(r *rand.Rand) float64 {
  val := 0.0
  for i := 0; i < int(e.k); i++ {
    val += exponential(r)
  }
  return val
}

func (e *Erlang) LogDensity(x float64) float64 {
  k64 := float64(e.k)
  return k64 * math.Log(e.lambda) + (k64 - 1.0) * math.Log(x) - e.lambda * x - lgamma(k64)
}

func (e *Erlang) Density(x float64) float64 {
  return math.Exp(e.LogDensity(x))
}
