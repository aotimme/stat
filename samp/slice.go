package samp

import (
  "math"
  "math/rand"

  "github.com/skelterjohn/go.matrix"
  "github.com/aotimme/stat/dist"
)

// Elliptical Slice Sampling
// See Algorithm 1 of http://arxiv.org/pdf/1210.7477v2.pdf
func EllipticalSlice(x []float64, logLik func([]float64) float64, mu []float64, sigma *matrix.DenseMatrix, r *rand.Rand) []float64 {
  p := len(x)
  norm := dist.NewMVNormal(mu, sigma)
  unif := dist.NewUniform(0, 1)
  angleUnif := dist.NewUniform(0, 2 * math.Pi)

  nu := norm.Sample(r)
  u := unif.Sample(r)

  theta := angleUnif.Sample(r)

  logy := logLik(x) + math.Log(u)
  for {
    thetaMin := theta - 2 * math.Pi
    thetaMax := theta
    cosTheta := math.Cos(theta)
    sinTheta := math.Cos(theta)

    xNew := make([]float64, p)
    for i := 0; i < p; i++ {
      xNew[i] = (x[i] - mu[i]) * cosTheta + (nu[i] - mu[i]) * sinTheta
    }
    if logLik(xNew) > logy {
      return xNew
    }
    if theta < 0 {
      thetaMin = theta
    } else {
      thetaMin = theta
    }
    angleUnif = dist.NewUniform(thetaMin, thetaMax)
    theta = angleUnif.Sample(r)
  }
}

// TODO
// * GeneralizedEllipticalSlice:
//     Algorightm 2 in http://arxiv.org/pdf/1210.7477v2.pdf
// * Metropolis
// * Metropolis-Hastings
