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

func exponential(r *rand.Rand) (n float64) {
  if r != nil {
    n = r.ExpFloat64()
  } else {
    n = rand.ExpFloat64()
  }
  return
}

// return log(X + Y) from log(X) and log(Y)
func logadd(logX, logY float64) float64 {
  // 1. make X the max
  if (logY > logX) {
    logX, logY = logY, logX
  }
  // 2. now X is bigger
  if (logX == math.Inf(-1)) {
    return logX
  }
  negDiff := logY - logX
  // 3. how far "down" (think decibels) is logY from logX?
  //    if it's really small (20 orders of magnitude smaller), then ignore
  if (negDiff < -20) {
    return logX
  }
  // 4. otherwise use some nice algebra to stay in the log domain
  //    (except for negDiff)
  return logX + math.Log(1.0 + math.Exp(negDiff))
}

// standard normal cdf function
func phi(x float64) float64 {
  return 0.5 * (1 + math.Erf(x / math.Sqrt(2)))
}

func RejectionSample(r *rand.Rand, targetLogPdf func(float64) float64, sourceLogPdf func(float64) float64, source func(*rand.Rand) float64, K float64) float64 {
  x := source(r)
  for math.Log(uniform(r)) >= targetLogPdf(x) - sourceLogPdf(x) - math.Log(K) {
    x = source(r)
  }
  return x
}
