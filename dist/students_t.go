package dist

import (
  "math"
  "math/rand"

  "github.com/aotimme/stat/utils"
)

type StudentsT struct {
  dof float64
  gam Gamma
}

func NewStudentsT(dof float64) (t StudentsT) {
  t.dof = dof
  t.gam = NewGamma(t.dof / 2.0, 1.0 / 2.0)
  return
}

func (t *StudentsT) Sample(r *rand.Rand) float64 {
  return stdnormal(r) / math.Sqrt(t.dof / t.gam.Sample(r))
}

func (t *StudentsT) LogDensity(x float64) float64 {
  a := utils.LogGamma((t.dof + 1) / 2)
  b := utils.LogGamma(t.dof / 2) + math.Log(math.Pi * t.dof) / 2
  c := (t.dof + 1) / 2 * math.Log(1 + x * x / t.dof)
  return a - b - c
}

func (t *StudentsT) Density(x float64) float64 {
  return math.Exp(t.LogDensity(x))
}
