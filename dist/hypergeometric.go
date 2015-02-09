package dist

import (
  "math"
  "math/rand"
)


type Hypergeometric struct {
  total, successes, n int64
}

func NewHypergeometric(total, successes, n int64) (h Hypergeometric) {
  h.total = total
  h.successes = successes
  h.n = n
  return
}

func (h *Hypergeometric) Sample(r *rand.Rand) (x int64) {
  return
}

func (h *Hypergeometric) LogDensity(x int64) float64 {
  return lchoose(h.successes, x) + lchoose(h.total - h.successes, h.n - x) - lchoose(h.total, h.n)
}

func (h *Hypergeometric) Density(x int64) float64 {
  return math.Exp(h.LogDensity(x))
}
