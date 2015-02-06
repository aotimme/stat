package dist

import (
  "math"
  "math/rand"
)

type InverseNormal struct {
  mu, lambda float64
}

func NewInverseNormal(mu, lambda float64) (in InverseNormal) {
  in.mu = mu
  in.lambda = lambda
  return
}

func (in *InverseNormal) LogDensity(x float64) float64 {
  z := 0.5 * (math.Ln2 + math.Log(math.Pi) + 3 * math.Log(x) - math.Log(in.lambda) )
  kern := in.lambda * (x - in.mu) * (x - in.mu) / (2 *  in.mu * in.mu * x)
  return kern - z
}

func (in *InverseNormal) CDF(x float64) float64 {
  sqrtlambx := math.Sqrt(in.lambda / x)
  e1 := phi(sqrtlambx * (x / in.mu - 1.0))
  e2 := math.Exp(2 * in.lambda / in.mu) * phi(-sqrtlambx * (x / in.mu + 1.0))
  return e1 + e2
}

// See: http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Generating_random_variates_from_an_inverse-Gaussian_distribution
func (in *InverseNormal) Sample(r *rand.Rand) float64 {
  nu := stdnormal(r)
  y := nu * nu
  x := in.mu + in.mu * in.mu * y / (2.0 * in.lambda) - in.mu / (2.0 * in.lambda) * math.Sqrt(4 * in.mu * in.lambda * y + in.mu * in.mu * y * y)
  z := uniform(r)
  if z <= in.mu / (in.mu + x) {
    return x
  } else {
    return in.mu * in.mu / x
  }
}
