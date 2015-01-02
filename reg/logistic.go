package reg

import (
  "fmt"
  "log"
  "math"
  "github.com/skelterjohn/go.matrix"
)

type Logistic struct {
  w0 float64
  intercept bool
  w []float64
  lambda float64
}

func NewLogistic(p int, intercept bool, lambda float64) (model Logistic) {
  model.w = make([]float64, p)
  if lambda >= 0.0 {
    model.lambda = lambda
  } else {
    model.lambda = 1e-6
  }
  model.intercept = intercept
  return
}

func NewLogisticWithData(data [][]float64, values []float64, intercept bool, lambda float64) (model Logistic, err error) {
  p := len(data[0])
  model = NewLogistic(p, intercept, lambda)
  err = model.Train(data, values)
  return
}

func (model *Logistic) Weights() []float64 {
  return model.w
}

func (model *Logistic) SetWeights(w []float64) {
  model.w = w
}

func (model *Logistic) HasIntercept() bool {
  return model.intercept
}

func (model *Logistic) Intercept() float64 {
  return model.w0
}

func (model *Logistic) SetIntercept(w0 float64) {
  model.intercept = true
  model.w0 = w0
}

func (model *Logistic) Train(data [][]float64, values []float64) (err error) {
  p := len(model.w)
  maxIterations := 100000
  iter := 0
  for {
    iter++
    rmse := 0.0

    grad := make([]float64, p)
    for j := 0; j < p; j++ {
      grad[j] = -model.w[j] * model.lambda
    }
    grad0 := 0.0

    var hess *matrix.DenseMatrix
    if model.intercept {
      hess = matrix.Eye(p + 1)
      hess.Scale(-model.lambda)
      hess.Set(p, p, 0.0)
    } else {
      hess = matrix.Eye(p)
      hess.Scale(model.lambda)
    }

    for i, x := range data {
      e := model.Predict(x)
      off := values[i] - e
      rmse += off * off
      for j, val := range x {
        grad[j] += off * val
      }
      for j := 0; j < p; j++ {
        for k := 0; k < p; k++ {
          hess.Set(j, k, hess.Get(j, k) + e * (1.0 - e) * x[j] * x[k])
        }
        if model.intercept {
          hess.Set(j, p, hess.Get(j, p) + e * (1.0 - e) * x[j])
          hess.Set(p, j, hess.Get(p, j) + e * (1.0 - e) * x[j])
          hess.Set(p, p, hess.Get(p, p) + e * (1.0 - e))
        }
      }
      if model.intercept {
        grad0 += off
      }
    }

    hessInv, err := hess.Inverse()
    if err != nil {
      fmt.Printf("ERROR (Inverse): %v\n", err)
      return err
    }

    var gradMat *matrix.DenseMatrix
    if model.intercept {
      gradMat = matrix.Zeros(p + 1, 1)
      gradMat.Set(p, 0, grad0)
    } else {
      gradMat = matrix.Zeros(p, 1)
    }
    for j := 0; j < p; j++ {
      gradMat.Set(j, 0, grad[j])
    }

    diffMat, err := hessInv.TimesDense(gradMat)
    if err != nil {
      return err
    }

    diff := 0.0
    for j := 0; j < p; j++ {
      add := diffMat.Get(j, 0)
      model.w[j] += add
      diff += add * add
    }
    if model.intercept {
      add := diffMat.Get(p, 0)
      model.w0 += add
      diff += add * add
    }

    if diff < 1e-6 {
      //log.Printf("Converged after %v iterations\n", iter)
      break
    }
    if iter >= maxIterations {
      //log.Printf("Did not converge after %v iterations\n", iter)
      break
    }
  }

  return
}

func (model *Logistic) Predict(datum []float64) float64 {
  value := dot(model.w, datum)
  if model.intercept {
    value += model.w0
  }
  return expit(value)
}

func (model *Logistic) RMSE(data [][]float64, values []float64) (rmse float64) {
  for i, datum := range data {
    rmse += math.Pow(values[i] - model.Predict(datum), 2.0)
  }
  rmse /= float64(len(data))
  return math.Sqrt(rmse)
}

func (model *Logistic) CV(data [][]float64, values []float64, fold int) (cv float64, err error) {
  if fold < 2 {
    fold = 5
  }
  // NOTE: stop if p > fold*n
  permData, permValues := permute(data, values, nil)
  numRun := 0
  for i := 0; i < fold; i++ {
    trainData, testData, trainValues, testValues := split(permData, permValues, i, fold)
    cvModel, err := NewLogisticWithData(trainData, trainValues, model.intercept, model.lambda)
    if err != nil {
      log.Printf("CV error: %v\n", err)
    } else {
      numRun++
    }
    cv += cvModel.RMSE(testData, testValues)
  }
  cv /= float64(numRun)
  return
}
