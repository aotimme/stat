package dist

import (
  "math"
  "math/rand"

  "github.com/skelterjohn/go.matrix"
)

type NormalInverseWishart struct {
  iwish InverseWishart
  mean []float64
  lambda float64
}

func NewNormalInverseWishart(mean []float64, lambda float64, sigma *matrix.DenseMatrix, nu int) (niw NormalInverseWishart) {
  niw.iwish = NewInverseWishart(sigma, nu)
  niw.mean = mean
  niw.lambda = lambda
  return
}

func (niw *NormalInverseWishart) Sample(r *rand.Rand) (mean []float64, cov *matrix.DenseMatrix) {
  cov = niw.iwish.Sample(r)
  covMean := cov.Copy()
  covMean.Scale(1.0 / niw.lambda)
  norm := NewMVNormal(niw.mean, covMean)
  mean = norm.Sample(r)
  return mean, cov
}

func (niw *NormalInverseWishart) LogDensity(mean []float64, cov *matrix.DenseMatrix) float64 {
  liw := niw.iwish.LogDensity(cov)
  // TODO: slow...
  covCopy := cov.Copy()
  mvn := NewMVNormal(mean, covCopy)
  return liw + mvn.LogDensity(mean)
}

func (niw *NormalInverseWishart) Density(mean []float64, cov *matrix.DenseMatrix) float64 {
  return math.Exp(niw.LogDensity(mean, cov))
}
