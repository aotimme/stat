package samp

import (
  "math"
  "math/rand"

  "github.com/aotimme/stat/dist"
)

func Rejection(targetLogPdf func(float64) float64, sourceLogPdf func(float64) float64, source func(*rand.Rand) float64, K float64, r *rand.Rand) float64 {
  x := source(r)
  u := dist.NewUniform(0, 1)
  for math.Log(u.Sample(r)) >= targetLogPdf(x) - sourceLogPdf(x) - math.Log(K) {
    x = source(r)
  }
  return x
}

func RejectionMV(targetLogPdf func([]float64) float64, sourceLogPdf func([]float64) float64, source func(*rand.Rand) []float64, K float64, r *rand.Rand) []float64 {
  x := source(r)
  u := dist.NewUniform(0, 1)
  for math.Log(u.Sample(r)) >= targetLogPdf(x) - sourceLogPdf(x) - math.Log(K) {
    x = source(r)
  }
  return x
}
