package dist

import (
  "math"
  "math/rand"

  "github.com/skelterjohn/go.matrix"
)

type MVStudentsT struct {
  dof float64
  mu []float64
  mvnorm MVNormal
  gam Gamma
}

func NewMVStudentsT(mu []float64, sigma *matrix.DenseMatrix, dof float64) (mvt MVStudentsT) {
  normMean := make([]float64, len(mu))
  mvt.mvnorm = NewMVNormal(normMean, sigma)
  mvt.mu = make([]float64, len(mu))
  copy(mvt.mu, mu)
  mvt.dof = dof
  mvt.gam = NewGamma(mvt.dof / 2, 1.0 / 2)
  return
}

func (mvt *MVStudentsT) Sample(r *rand.Rand) (s []float64) {
  y := mvt.mvnorm.Sample(r)
  x := mvt.gam.Sample(r)
  p := len(mvt.mu)
  s = make([]float64, p)
  copy(s, mvt.mu)
  for i := 0; i < p; i++ {
    s[i] += y[i] * math.Sqrt(mvt.dof / x)
  }
  return
}

func (mvt *MVStudentsT) LogDensity(x []float64) float64 {
  p := len(mvt.mu)
  pf := float64(p)
  quad := 0.0
  covInv := mvt.mvnorm.getInverseCovariance()
  for i := 0; i < p; i++ {
    for j := 0; j < p; j++ {
      quad += (x[i] - mvt.mu[i]) * covInv.Get(i, j) * (x[j] - mvt.mu[j])
    }
  }
  norm := lgamma((mvt.dof + pf) / 2) - lgamma(mvt.dof / 2) - pf * math.Log(mvt.dof) / 2 - pf * math.Log(math.Pi) / 2 - mvt.mvnorm.getLogDetCov() / 2
  return norm - (mvt.dof + pf) / 2 * math.Log(1.0 + quad / mvt.dof)
}

func (mvt *MVStudentsT) Density(x []float64) float64 {
  return math.Exp(mvt.LogDensity(x))
}
