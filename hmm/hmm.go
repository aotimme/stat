package hmm

import (
  "fmt"
  "math"
  "math/rand"
)

func ForwardBackward(y []int, P, E [][]float64, nu []float64) (a, b [][]float64, c []float64) {
  K := len(P)
  N := len(y)

  a = make([][]float64, N)
  b = make([][]float64, N)
  c = make([]float64, N)
  for t := 0; t < N; t++ {
    a[t] = make([]float64, K)
    b[t] = make([]float64, K)
  }

  // forward
  for k := 0; k < K; k++ {
    atk := nu[k] * E[k][y[0]]
    a[0][k] = atk
    c[0] += atk
  }
  for k := 0; k < K; k++ {
    a[0][k] /= c[0]
  }
  for t := 0; t < N-1; t++ {
    for k := 0; k < K; k++ {
      sum := 0.0
      for j := 0; j < K; j++ {
        sum += a[t][j] * P[j][k]
      }
      atk := E[k][y[t+1]] * sum
      a[t+1][k] = atk
      c[t+1] += atk
    }
    for k := 0; k < K; k++ {
      a[t+1][k] /= c[t+1]
    }
  }

  // backward
  for k := 0; k < K; k++ {
    b[N-1][k] = 1.0
  }
  for t := N-2; t >= 0; t-- {
    for k := 0; k < K; k++ {
      sum := 0.0
      for j := 0; j < K; j++ {
        sum += P[k][j] * E[j][y[t+1]] * b[t+1][j]
      }
      b[t][k] = sum / c[t+1]
    }
  }
  return
}

func BaumWelch(y []int, P0, E0 [][]float64, nu0 []float64) (P, E [][]float64, nu []float64, logprobs []float64) {
  epsilon := 0.01

  N := len(y)       // sequence length
  K := len(P0)      // number of latent states
  S := len(E0[0])   // number of emitted states

  // initialize parameters
  P = make([][]float64, K)
  E = make([][]float64, K)
  nu = make([]float64, K)
  for k := 0; k < K; k++ {
    P[k] = make([]float64, K)
    E[k] = make([]float64, S)
    copy(P[k], P0[k])
    copy(E[k], E0[k])
  }
  copy(nu, nu0)

  prevLogprob := 0.0
  logprobs = make([]float64, 1)
  iter := 0
  for {
    a, b, c := ForwardBackward(y, P, E, nu)
    logprob := 0.0
    for _, cc := range c {
      logprob += math.Log(cc)
    }
    fmt.Printf("%v\n", logprob)
    if iter == 0 {
      logprobs[iter] = logprob
    } else {
      logprobs = append(logprobs, logprob)
    }
    if math.Abs(logprob - prevLogprob) < epsilon {
      break
    }
    prevLogprob = logprob

    // Computing nu
    for k := 0; k < K; k++ {
      nu[k] = a[0][k] * b[0][k]
    }
    // Computing P
    for i := 0; i < K; i++ {
      sumpij := 0.0
      for j := 0; j < K; j++ {
        pij := 0.0
        for t := 1; t < N; t++ {
          pij += a[t-1][i] * E[j][y[t]] * P[i][j] * b[t][j] / c[t]
        }
        sumpij += pij
        P[i][j] = pij
      }
      for j := 0; j < K; j++ {
        P[i][j] /= sumpij
      }
    }
    // Computing E
    for k := 0; k < K; k++ {
      for s := 0; s < S; s++ {
        E[k][s] = 0.0
      }
    }
    for t := 0; t < N; t++ {
      s := y[t]
      for k := 0; k < K; k++ {
        E[k][s] += a[t][k] * b[t][k]
      }
    }
    for k := 0; k < K; k++ {
      sum := 0.0
      for s := 0; s < S; s++ {
        sum += E[k][s]
      }
      for s := 0; s < S; s++ {
        E[k][s] /= sum
      }
    }
    iter++
  }

  return
}

func Viterbi(y []int, P, E [][]float64, nu []float64) (vit []int) {
  N := len(y)
  K := len(P)

  d := make([][]float64, N)
  for t := 0; t < N; t++ {
    d[t] = make([]float64, K)
  }
  f := make([][]int, N)
  for t := 0; t < N; t++ {
    f[t] = make([]int, K)
  }

  for k := 0; k < K; k++ {
    d[0][k] = math.Log(nu[k]) + math.Log(E[k][y[0]])
  }
  for t := 1; t < N; t++ {
    for k := 0; k < K; k++ {
      maxlogprob := d[t-1][0] + math.Log(P[0][k]) + math.Log(E[0][y[t]])
      whichmax := 0
      for j := 1; j < K; j++ {
        logprob := d[t-1][j] + math.Log(P[j][k]) + math.Log(E[j][y[t]])
        if logprob > maxlogprob {
          maxlogprob = logprob
          whichmax = j
        }
      }
      d[t][k] = maxlogprob
      f[t][k] = whichmax
    }
  }

  vit = make([]int, N)
  curmax := 0
  for k := 1; k < K; k++ {
    if d[N-1][k] > d[N-1][curmax] {
      curmax = k
    }
  }
  vit[N-1] = curmax
  for t := N-2; t >= 0; t-- {
    vit[t] = f[t+1][vit[t+1]]
  }

  return
}

func draw(p []float64, r *rand.Rand) (z int) {
  var u float64
  if r != nil {
    u = r.Float64()
  } else {
    u = rand.Float64()
  }
  sum := p[z]
  for sum < u {
    z++
    sum += p[z]
  }
  return
}

func GenerateSequence(N int, P, E [][]float64, nu []float64, r *rand.Rand) (hidden []int, observed []int) {
  z := make([]int, N)
  y := make([]int, N)

  z[0] = draw(nu, r)
  y[0] = draw(E[z[0]], r)
  for t := 0; t < N-1; t++ {
    z[t+1] = draw(P[z[t]], r)
    y[t+1] = draw(E[z[t+1]], r)
  }

  hidden = z
  observed = y

  return
}
