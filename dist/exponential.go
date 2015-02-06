package dist

import (
  "math/rand"
  "math"
)

type Exponential struct {
  lambda float64
}

func NewExponential(lambda float64) (exp Exponential) {
  exp.lambda = lambda
  return
}

func (exp *Exponential) Sample(r *rand.Rand) float64 {
  s := exponential(r)
  return s / exp.lambda
}

func (exp *Exponential) LogDensity(x float64) float64 {
  return math.Log(exp.lambda) - exp.lambda * x
}

func (exp *Exponential) Density(x float64) float64 {
  return math.Exp(exp.LogDensity(x))
}
