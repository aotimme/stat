package dist

import (
  "math"
  "math/rand"
)

type Logarithmic struct {
  p float64
}

func NewLogarithmic(p float64) (l Logarithmic) {
  l.p = p
  return
}

// From: https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/distributions.c
func (l *Logarithmic) Sample(r *rand.Rand) int64 {
  rn := math.Log(1.0 - l.p)
  for {
    v := uniform(r)
    if v >= l.p {
      return 1
    }
    u := uniform(r)
    q := 1.0 - math.Exp(rn * u)
    if v < q * q {
      res := int64(1.0 + math.Log(v) / math.Log(q))
      if res < 1 {
        continue
      } else {
        return res
      }
    }
    if (v >= q) {
      return 1
    }
    return 2
  }
}

func (l *Logarithmic) LogDensity(k int64) float64 {
  k64 := float64(k)
  return k64 * math.Log(l.p) - math.Log(k64) + math.Log(1.0 / (1.0 - l.p))
}

func (l *Logarithmic) Density(k int64) float64 {
  return math.Exp(l.LogDensity(k))
}
