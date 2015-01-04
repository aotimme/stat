package dist

import (
  "math"
  "math/rand"
)

type Normal struct {
  mean float64
  variance float64
}

func NewNormal(mean, variance float64) (n Normal) {
  n.mean = mean
  n.variance = variance
  return
}

func (n *Normal) Sample(r *rand.Rand) float64 {
  return stdnormal(r) * math.Sqrt(n.variance) + n.mean
}

func (n *Normal) LogDensity(x float64) float64 {
  quad := (x - n.mean) * (x - n.mean)
  return -0.5 * (quad / n.variance + math.Log(2 * math.Pi * n.variance))
}

func (n *Normal) Density(x float64) float64 {
  return math.Exp(n.LogDensity(x))
}
