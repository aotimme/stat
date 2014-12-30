package stat

import (
  "math/rand"
  "fmt"
)

type Categorical struct {
  p []float64
}

func NewCategorical(p []float64) (cat Categorical) {
  cat.p = p
  sum := 0.0
  for _, elem := range p {
    sum += elem
  }
  if sum != 1.0 {
    for i := 0; i < len(p); i++ {
      cat.p[i] /= sum
    }
  }
  return
}

func (cat *Categorical) Sample(r *rand.Rand) (z int64) {
  u := NextUniform(r)
  sum := cat.p[z]
  for sum < u {
    z++
    sum += cat.p[z]
  }
  if z == int64(len(cat.p)) {
    fmt.Printf("UH-OH! u = %v\n", u)
  }
  return
}
