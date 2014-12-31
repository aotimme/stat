package dist

import (
  "math"
  "math/rand"
  "fmt"

  "github.com/skelterjohn/go.matrix"
)

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

func (iwish *InverseWishart) LogDensity(X *matrix.DenseMatrix) float64 {
  Xinv, err := X.Inverse()
  if err != nil {
    // could also return -Inf...
    panic(err)
  }
  return iwish.wish.LogDensity(Xinv)
}

func (iwish *InverseWishart) Density(X *matrix.DenseMatrix) float64 {
  return math.Exp(iwish.LogDensity(X))
}
