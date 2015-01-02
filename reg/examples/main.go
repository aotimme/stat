package main

import (
  "encoding/csv"
  "fmt"
  "os"
  "strconv"

  "github.com/aotimme/stat/reg"
)

func parseFloatOrDie(elem string) (f float64) {
  f, err := strconv.ParseFloat(elem, 64)
  if err != nil {
    panic(err)
  }
  return
}

func GetSwissData() (data [][]float64, values []float64) {
  filename := "swiss.csv"
  f, err := os.Open(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()
  r := csv.NewReader(f)
  records, err := r.ReadAll()
  if err != nil {
    panic(err)
  }

  n := len(records) - 1
  p := len(records[0]) - 2
  data = make([][]float64, n)
  values = make([]float64, n)
  for i := 0; i < n; i++ {
    data[i] = make([]float64, p)
  }
  for i, line := range records {
    if i == 0 {
      continue
    }
    for j, elem := range line {
      if j == 0 {
        continue
      }
      num := parseFloatOrDie(elem)
      if j == 1 {
        values[i-1] = num
      } else {
        data[i-1][j-2] = num
      }
    }
  }
  return
}

func RunLinear() {
  data, values := GetSwissData()
  model, err := reg.NewLinearWithData(data, values, true, 0.1)
  if err != nil {
    panic(err)
  }

  w := model.Weights()
  w0 := model.Intercept()

  // To check in R:
  // > lm(Fertility ~ ., data=swiss)
  fmt.Printf("Linear:\n")
  fmt.Printf("  w0: %v\n", w0)
  fmt.Printf("  w: %v\n", w)
}


func GetMenarcheData() (data [][]float64, values []float64) {
  filename := "menarche.csv"
  f, err := os.Open(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()
  r := csv.NewReader(f)
  records, err := r.ReadAll()
  if err != nil {
    panic(err)
  }

  n := len(records) - 1
  data = make([][]float64, n)
  values = make([]float64, n)
  for i := 0; i < n; i++ {
    data[i] = make([]float64, 1)
  }
  for i, line := range records {
    if i == 0 {
      continue
    }
    data[i-1][0] = parseFloatOrDie(line[1])
    menarche := parseFloatOrDie(line[3])
    total := parseFloatOrDie(line[2])
    values[i-1] = menarche / total
  }
  return
}

func RunLogistic() {
  data, values := GetMenarcheData()
  model, err := reg.NewLogisticWithData(data, values, true, 0.0)
  if err != nil {
    panic(err)
  }

  w := model.Weights()
  w0 := model.Intercept()

  // To check in R:
  // > library(MASS)
  // > glm(cbind(Menarche, Total-Menarche) ~ Age, family=binomial(logit), data=menarche)
  fmt.Printf("Logistic:\n")
  fmt.Printf("  w0: %v\n", w0)
  fmt.Printf("  w: %v\n", w)
}

func main() {
  RunLinear()
  RunLogistic()
}
