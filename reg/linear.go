package reg

import (
  "log"
  "math"
  "github.com/skelterjohn/go.matrix"
)

type Linear struct {
  w0 float64
  intercept bool
  w []float64
  lambda float64
}

func NewLinear(p int, intercept bool, lambda float64) (model Linear) {
  model.w = make([]float64, p)
  if lambda >= 0.0 {
    model.lambda = lambda
  } else {
    model.lambda = 1e-6
  }
  model.intercept = intercept
  return
}

func NewLinearWithData(data [][]float64, values []float64, intercept bool, lambda float64) (model Linear, err error) {
  p := len(data[0])
  model = NewLinear(p, intercept, lambda)
  err = model.Train(data, values)
  return
}

func (model *Linear) Weights() []float64 {
  return model.w
}

func (model *Linear) SetWeights(w []float64) {
  model.w = w
}

func (model *Linear) HasIntercept() bool {
  return model.intercept
}

func (model *Linear) Intercept() float64 {
  return model.w0
}

func (model *Linear) SetIntercept(w0 float64) {
  model.intercept = true
  model.w0 = w0
}

func (model *Linear) Train(data [][]float64, values []float64) (err error) {
  n := len(data)
  p := len(model.w)
  if model.intercept {
    p++
    for i, _ := range data {
      data[i] = append(data[i], 1.0)
    }
  }
  X := matrix.MakeDenseMatrixStacked(data)
  y := matrix.MakeDenseMatrix(values, n, 1)
  Xt := X.Transpose()
  XtX, err := Xt.TimesDense(X)
  if err != nil {
    return
  }
  for j := 0; j < p-1; j++ {
    XtX.Set(j, j, XtX.Get(j, j) + model.lambda)
  }
  if !model.intercept {
    XtX.Set(p-1, p-1, XtX.Get(p-1, p-1) + model.lambda)
  }

  XtXInv, err := XtX.Inverse()
  if err != nil {
    return
  }
  XtY, err := Xt.TimesDense(y)
  if err != nil {
    return
  }
  coefficients, err := XtXInv.TimesDense(XtY)
  if err != nil {
    return
  }
  coefs := coefficients.Array()
  if model.intercept {
    model.w0 = coefs[p-1]
    model.w = coefs[:p-1]
  } else {
    model.w = coefs
  }
  return
}

func (model *Linear) Predict(datum []float64) (value float64) {
  value = dot(model.w, datum)
  if model.intercept {
    value += model.w0
  }
  return
}

func (model *Linear) RMSE(data [][]float64, values []float64) float64 {
  rmse := 0.0
  for i, x := range data {
    rmse += math.Pow(values[i] - model.Predict(x), 2.0)
  }
  rmse /= float64(len(data))
  return math.Sqrt(rmse)
}

func (model *Linear) CV(data [][]float64, values []float64, fold int) (cv float64, err error) {
  if fold < 2 {
    fold = 5
  }
  // NOTE: stop if p > fold*n
  permData, permValues := permute(data, values, nil)
  numRun := 0
  for i := 0; i < fold; i++ {
    trainData, testData, trainValues, testValues := split(permData, permValues, i, fold)
    cvModel, err := NewLinearWithData(trainData, trainValues, model.intercept, model.lambda)
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
