package reg

import (
  "math"
  "math/rand"
)

func dot(vec1, vec2 []float64) (val float64) {
  for i, v := range vec1 {
    val += v * vec2[i]
  }
  return
}

func expit(val float64) float64 {
  return 1.0 / (1.0 + math.Exp(-val))
}

func permute(data [][]float64, values []float64, r *rand.Rand) (permData [][]float64, permValues []float64) {
  n := len(data)
  p := len(data[0])
  var perm []int
  if r != nil {
    perm = r.Perm(n)
  } else {
    perm = rand.Perm(n)
  }
  permData = make([][]float64, n)
  permValues = make([]float64, n)
  for i := 0; i < n; i++ {
    permData[i] = make([]float64, p)
    copy(permData[i], data[i])
  }

  for i, j := range perm {
    permData[i], permData[j] = permData[j], permData[i]
    permValues[i], permValues[j] = permValues[j], permValues[i]
  }
  return
}

func split(data [][]float64, values []float64, fold, nFold int) (trainData, testData [][]float64, trainValues, testValues []float64) {
  n := len(data)
  numPer := n / nFold
  mod := n % nFold

  minBreakVal := 0
  for j := 0; j < fold; j++ {
    minBreakVal += numPer
    if j < mod {
      minBreakVal++
    }
  }
  maxBreakVal := 0
  for j := 0; j < fold + 1; j++ {
    maxBreakVal += numPer
    if j < mod {
      maxBreakVal++
    }
  }
  num := maxBreakVal - minBreakVal
  trainValues = make([]float64, n - num)
  trainData = make([][]float64, n - num)
  testValues = make([]float64, num)
  testData = make([][]float64, num)
  for j := 0; j < n; j++ {
    if j < minBreakVal {
      trainData[j] = data[j]
      trainValues[j] = values[j]
    } else if j < maxBreakVal {
      testData[j - minBreakVal] = data[j]
      testValues[j - minBreakVal] = values[j]
    } else {
      trainData[j - num] = data[j]
      trainValues[j - num] = values[j]
    }
  }
  return
}
