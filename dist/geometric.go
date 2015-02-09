package dist

import (
  "math"
  "math/rand"
)

// Geometric Distribution
// RV is the number of trials until a success (1, 2, 3, ...)
type Geometric struct {
  p float64
}

func NewGeometric(p float64) (g Geometric) {
  g.p = p
  return
}

// starts at 1
func (g *Geometric) Sample(r *rand.Rand) (k int64) {
  for {
    k++
    if uniform(r) < g.p {
      break
    }
  }
  return
}

func (g *Geometric) LogDensity(k int64) float64 {
  return float64(k - 1) * math.Log(1 - g.p) + math.Log(g.p)
}

func (g *Geometric) Density(k int64) float64 {
  return math.Exp(g.LogDensity(k))
}
