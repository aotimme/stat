package clust

import (
  "math"

  "github.com/aotimme/stat/dist"
  "github.com/skelterjohn/go.matrix"
)

type hyperparameters struct {
  mu0 []float64
  lambda0 float64
  sigma0 *matrix.DenseMatrix
  nu0 int
  alpha0 []float64
}

func newHyperparameters(p, k int) (self hyperparameters) {
  self.mu0 = make([]float64, p)
  diag := make([]float64, p)
  for i := 0; i < p; i++ {
    diag[i] = float64(p)
  }
  self.sigma0 = matrix.Diagonal(diag)
  self.lambda0 = 1.0
  self.nu0 = p
  self.alpha0 = make([]float64, k)
  for c := 0; c < k; c++ {
    self.alpha0[c] = 0.1
  }
  return
}

type suffStats struct {
  nk []float64
  sumX [][]float64
  sumXXt []*matrix.DenseMatrix
}

func newSuffStats(p, k int) (stats suffStats) {
  stats.nk = make([]float64, k)
  stats.sumX = make([][]float64, k)
  stats.sumXXt = make([]*matrix.DenseMatrix, k)
  for kk := 0; kk < k; kk++ {
    stats.sumX[kk] = make([]float64, p)
    stats.sumXXt[kk] = matrix.Zeros(p, p)
  }
  return
}

func (ss *suffStats) AddPoint(x []float64, probs []float64) {
  k := len(ss.sumX)
  p := len(x)

  for c := 0; c < k; c++ {
    pr := probs[c]
    ss.nk[c] += pr
    for i := 0; i < p; i++ {
      ss.sumX[c][i] += pr * x[i]
      for j := i; j < p; j++ {
        elem := ss.sumXXt[c].Get(i, j) + pr * x[i] * x[j]
        ss.sumXXt[c].Set(i, j, elem)
        if j > i {
          ss.sumXXt[c].Set(j, i, elem)
        }
      }
    }
  }
}

type GMM struct {
  k int
  hyp hyperparameters

  mus [][]float64
  sigmas []*matrix.DenseMatrix
  pi []float64
  gaussians []dist.MVNormal
  probs [][]float64
}

func NewGMM(k int) (self GMM) {
  self.k = k
  return
}

func (self *GMM) Initialize(X [][]float64) (stats suffStats) {
  km := NewKMeans(self.k)
  z := km.Initialize(X, nil)

  p := len(X[0])
  stats = newSuffStats(p, self.k)

  self.probs = make([][]float64, len(X))
  for i, x := range X {
    probs := make([]float64, p)
    probs[z[i]] = 1.0
    stats.AddPoint(x, probs)
    self.probs[i] = probs
  }
  return
}

func (self *GMM) UpdateMAP(stats suffStats) {
  k := self.k
  p := len(stats.sumX[0])

  self.mus = make([][]float64, k)
  self.sigmas = make([]*matrix.DenseMatrix, k)
  self.pi = make([]float64, k)
  self.gaussians = make([]dist.MVNormal, k)
  sumpi := 0.0
  for c := 0; c < k; c++ {
    self.pi[c] = stats.nk[c] - 1.0
    sumpi += self.pi[c]
    self.mus[c] = make([]float64, p)
    for i := 0; i < p; i++ {
      self.mus[c][i] = (stats.sumX[c][i] + self.hyp.mu0[i] * self.hyp.lambda0) / (stats.nk[c] + self.hyp.lambda0)
    }
    self.sigmas[c] = matrix.Zeros(p, p)
    for i := 0; i < p; i++ {
      for j := 0; j < p; j++ {
        elem := self.hyp.sigma0.Get(i, j) + self.hyp.lambda0 * self.hyp.mu0[i] * self.hyp.mu0[j] + stats.sumXXt[c].Get(i, j) - (stats.nk[c] + self.hyp.lambda0) * self.mus[c][i] * self.mus[c][j]
        self.sigmas[c].Set(i, j, elem)
        if j != i {
          self.sigmas[c].Set(j, i, elem)
        }
      }
    }
    self.sigmas[c].Scale(1.0 / (stats.nk[c] + float64(self.hyp.nu0 + p + 1)))
    self.gaussians[c] = dist.NewMVNormal(self.mus[c], self.sigmas[c])
  }
  for c := 0; c < k; c++ {
    self.pi[c] /= sumpi
  }
}

func (self *GMM) UpdateMLE(stats suffStats) {
  k := self.k
  p := len(stats.sumX[0])

  self.mus = make([][]float64, k)
  self.sigmas = make([]*matrix.DenseMatrix, k)
  self.pi = make([]float64, k)
  self.gaussians = make([]dist.MVNormal, k)
  sumpi := 0.0
  for c := 0; c < k; c++ {
    self.pi[c] = stats.nk[c]
    sumpi += self.pi[c]
    self.mus[c] = make([]float64, p)
    for i := 0; i < p; i++ {
      self.mus[c][i] = stats.sumX[c][i] / stats.nk[c]
    }
    self.sigmas[c] = matrix.Zeros(p, p)
    for i := 0; i < p; i++ {
      for j := 0; j < p; j++ {
        elem := stats.sumXXt[c].Get(i, j) / stats.nk[c] - self.mus[c][i] * self.mus[c][j]
        self.sigmas[c].Set(i, j, elem)
        if j != i {
          self.sigmas[c].Set(j, i, elem)
        }
      }
    }
    self.gaussians[c] = dist.NewMVNormal(self.mus[c], self.sigmas[c])
  }
  for c := 0; c < k; c++ {
    self.pi[c] /= sumpi
  }
}


func (self *GMM) GetProbs(x []float64) (probs []float64) {
  p := len(x)
  k := self.k

  probs = make([]float64, p)
  sumprobs := 0.0
  for c := 0; c < k; c++ {
    probs[c] = math.Exp(self.gaussians[c].LogDensity(x) + math.Log(self.pi[c]))
    sumprobs += probs[c]
  }
  for c := 0; c < k; c++ {
    probs[c] /= sumprobs
  }
  return
}

func (self *GMM) cluster(X [][]float64, asMap bool) (mus [][]float64, sigmas []*matrix.DenseMatrix, pi []float64) {
  p := len(X[0])
  self.hyp = newHyperparameters(p, self.k)

  maxIters := 200

  stats := self.Initialize(X)
  if asMap {
    self.UpdateMAP(stats)
  } else {
    self.UpdateMLE(stats)
  }

  for iter := 0; iter < maxIters; iter++ {
    stats = newSuffStats(p, self.k)
    for i, x := range X {
      probs := self.GetProbs(x)
      stats.AddPoint(x, probs)
      self.probs[i] = probs
    }
    if asMap {
      self.UpdateMAP(stats)
    } else {
      self.UpdateMLE(stats)
    }
  }

  return self.mus, self.sigmas, self.pi
}

func (self *GMM) ClusterMLE(X [][]float64) (mus [][]float64, sigmas []*matrix.DenseMatrix, pi []float64) {
  return self.cluster(X, false)
}

func (self *GMM) ClusterMAP(X [][]float64) (mus [][]float64, sigmas []*matrix.DenseMatrix, pi []float64) {
  return self.cluster(X, true)
}

func (self *GMM) Memberships(idx int) (probs []float64) {
  p := len(self.mus[0])
  probs = make([]float64, p)
  copy(probs, self.probs[idx])
  return
}

func (self *GMM) Means() [][]float64 {
  return self.mus
}
func (self *GMM) Mean(idx int) []float64 {
  return self.mus[idx]
}
func (self *GMM) Covariances() []*matrix.DenseMatrix {
  return self.sigmas
}
func (self *GMM) Covariance(idx int) *matrix.DenseMatrix {
  return self.sigmas[idx]
}
func (self *GMM) Proportions() []float64 {
  return self.pi
}
func (self *GMM) Proportion(idx int) float64 {
  return self.pi[idx]
}
