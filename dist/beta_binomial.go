package dist

import (
  "math"
  "math/rand"
)

type BetaBinomial struct {
  beta *Beta
  a, b float64
  n int64
}

func NewBetaBinomial(n int64, a, b float64) (bb BetaBinomial) {
  bb.a, bb.b, bb.n = a, b, n
  beta := NewBeta(a, b)
  bb.beta = &beta
  return
}

func (bb *BetaBinomial) Sample(r *rand.Rand) int64 {
  p := bb.beta.Sample(r)
  bin := NewBinomial(bb.n, p)
  return bin.Sample(r)
}

func (bb *BetaBinomial) LogDensity(k int64) float64 {
  n64, k64 := float64(bb.n), float64(k)
  return lchoose(bb.n, k) + lbeta(k64 + bb.a, n64 - k64 + bb.b) - lbeta(bb.a, bb.b)
}

func (bb *BetaBinomial) Density(x int64) float64 {
  return math.Exp(bb.LogDensity(x))
}
