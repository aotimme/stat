package dist

import (
  "math"
  "math/rand"
)

func uniform(r *rand.Rand) (u float64) {
  if r != nil {
    u = r.Float64()
  } else {
    u = rand.Float64()
  }
  return
}

func stdnormal(r *rand.Rand) (n float64) {
  if r != nil {
    n = r.NormFloat64()
  } else {
    n = rand.NormFloat64()
  }
  return
}

func RejectionSample(r *rand.Rand, targetLogPdf func(float64) float64, sourceLogPdf func(float64) float64, source func(*rand.Rand) float64, K float64) float64 {
  x := source(r)
  for math.Log(uniform(r)) >= targetLogPdf(x) - sourceLogPdf(x) - math.Log(K) {
    x = source(r)
  }
  return x
}
