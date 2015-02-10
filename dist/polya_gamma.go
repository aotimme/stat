package dist

import (
  "math"
  "math/rand"
)

// See: http://arxiv.org/pdf/1205.0310v3.pdf
type PolyaGamma struct {
  b, z float64
}

func NewPolyaGamma(b, z float64) (pg PolyaGamma) {
  pg.b = b
  pg.z = z
  return
}

func pigauss(t, mu, lambda float64) float64 {
  in := NewInverseNormal(mu, lambda)
  return in.CDF(t)
}

func loganx(n int, x, t float64) float64 {
  nhalf := float64(n) + 0.5
  logshared := math.Log(math.Pi) + math.Log(nhalf)
  if x <= t {
    return logshared + 1.5 * (math.Log(2.0) - math.Log(math.Pi) - math.Log(x)) - 2.0 * nhalf * nhalf / x
  } else {
    return logshared - nhalf * nhalf * math.Pi * math.Pi * x / 2.0
  }
}

func anx(n int, x, t float64) float64 {
  return math.Exp(loganx(n, x, t))
}

func (pg *PolyaGamma) Sample(r *rand.Rand) float64 {
  // Algorithm 1 in the http://arxiv.org/pdf/1205.0310v3.pdf Supplement
  z := math.Abs(pg.z) / 2.0
  t := 0.64
  K := math.Pi * math.Pi / 8.0 + z * z / 2.0
  p := math.Pi / (2.0 * K) * math.Exp(-K * t)
  q := 2 * math.Exp(-z) * pigauss(t, 1.0 / z, 1.0)
  for {
    u := uniform(r)
    v := uniform(r)
    var x float64
    if u < p / (p + q) {
      // truncated exponential
      x = t + exponential(r) / K
    } else {
      // truncated inverse gaussian
      mu := 1.0 / z
      if mu > t {
        // Algorithm 2 in the http://arxiv.org/pdf/1205.0310v3.pdf Supplement
        for {
          var e1, e2 float64
          for {
            e1 = exponential(r)
            e2 = exponential(r)
            if e1 * e1 <= 2 * e2 / t {
              break
            }
          }
          x = t / (1.0 + t * e1) * (1.0 + t * e1)
          if math.Log(uniform(r)) <= -0.5 * z * z * x {
            break
          }
        }
      } else {
        in := NewInverseNormal(mu, 1.0)
        for {
          x = in.Sample(r)
          if x < t {
            break
          }
        }
      }
    }
    // now we have an X -- time to accept/reject
    logS := loganx(0, x, t)
    logY := math.Log(v) + logS
    n := 0
    for {
      n++
      if n % 2 == 1 {
        // n is odd
        logS = logadd(logS, loganx(n, x, t))
        if logY < logS {
          return 0.25 * x
        }
      } else {
        // n is even
        logS = logadd(logS, loganx(n, x, t))
        if logY > logS {
          break
        }
      }
    }
  }
}

func (pg *PolyaGamma) SampleTruncatedSum(truncation int, r *rand.Rand) float64 {
  gam := NewGamma(pg.b, 1.0)
  x := 0.0
  pisq := math.Pi * math.Pi
  zsq := pg.z * pg.z
  for k := 0; k < truncation; k++ {
    k64 := float64(k + 1)
    gk := gam.Sample(r)
    x += gk / (math.Pow(k64 - 0.5, 2.0) + zsq / (4.0 * pisq))
  }
  return x / (2.0 * pisq)
}

// No LogDensity...
