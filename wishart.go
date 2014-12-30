package stat

import (
  "github.com/skelterjohn/go.matrix"
  //"github.com/gonum/matrix/mat64"
  //"errors"
  "math/rand"
  "fmt"
)


type Wishart struct {
  norm Normal
  nu int
}

func NewWishart(sigma *matrix.DenseMatrix, nu int) (wish Wishart) {
  p := sigma.Rows()
  wish.norm = NewNormal(make([]float64, p), sigma)
  wish.nu = nu
  return
}

func (wish *Wishart) Sample(r *rand.Rand) (sigma *matrix.DenseMatrix) {
  X := wish.norm.SampleMultiple(wish.nu, r)
  XT := X.Transpose()
  sigma, err := XT.TimesDense(X)
  if err != nil {
    panic(err)
  }
  p := sigma.Rows()
  for i := 0; i < p; i++ {
    for j := 0; j < p; j++ {
      if sigma.Get(i, j) != sigma.Get(j, i) {
        fmt.Printf("sigma (i,j) != (j,i)\n")
      }
    }
  }
  return
}

type InverseWishart struct {
  wish Wishart
}

func NewInverseWishart(sigma *matrix.DenseMatrix, nu int) (iwish InverseWishart) {
  sigmaInv, err := sigma.Inverse()
  if err != nil {
    panic(err)
  }

  // ensure symmetry (inverse doesn't always maintain the symmetry, although max difference 1e-18, so close...)
  p := sigmaInv.Rows()
  for i := 0; i < p; i++ {
    for j := i+1; j < p; j++ {
      if sigmaInv.Get(i, j) != sigmaInv.Get(j, i) {
        sigmaInv.Set(j, i, sigmaInv.Get(i, j))
      }
    }
  }

  iwish.wish = NewWishart(sigmaInv, nu)
  return
}

func (iwish *InverseWishart) Sample(r *rand.Rand) *matrix.DenseMatrix {
  s := iwish.wish.Sample(r)
  inv, err := s.Inverse()

  if err != nil {
    fmt.Printf("Uh-oh. Matrix = %v\n", s)
    fmt.Printf("nu = %v\n", iwish.wish.nu)
    fmt.Printf("cov = %v\n", iwish.wish.norm.cov)
    fmt.Printf("det(cov) = %v\n", iwish.wish.norm.cov.Det())
    panic(err)
  }

  // ensure symmetry (inverse doesn't always maintain the symmetry, although max difference 1e-18, so close...)
  p := inv.Rows()
  for i := 0; i < p; i++ {
    for j := i+1; j < p; j++ {
      if inv.Get(i, j) != inv.Get(j, i) {
        inv.Set(j, i, inv.Get(i, j))
      }
    }
  }

  return inv
}

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
  norm := NewNormal(niw.mean, covMean)
  mean = norm.Sample(r)
  return mean, cov
}
