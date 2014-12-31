package main

// Example usage of "hmm" package, using wind direction data from
// Snoqualmie Falls

import (
  "fmt"
  "io/ioutil"
  "math/rand"
  "os"
  "strconv"
  "strings"

  "github.com/aotimme/stat/hmm"
)

func GetData(filename string) (y []int) {
  data, err := ioutil.ReadFile(filename)
  if err != nil {
    panic(err)
  }
  str := string(data)
  str = strings.Replace(str, "\n", " ", -1)
  strs := strings.Split(str, " ")
  numInts := 0
  for _, s := range strs {
    if _, err = strconv.Atoi(s); err == nil {
      numInts++
    }
  }
  y = make([]int, numInts)
  idx := 0
  for _, s := range strs {
    curInt, err := strconv.Atoi(s)
    if err == nil {
      y[idx] = curInt-1
      idx++
    }
  }
  return
}

func WriteMatrix(filename string, M [][]float64) {
  f, err := os.Create(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()
  strs := make([]string, len(M[0]))
  for i := 0; i < len(M); i++ {
    for j := 0; j < len(M[i]); j++ {
      strs[j] = fmt.Sprintf("%f", M[i][j])
    }
    f.WriteString(fmt.Sprintf("%s\n", strings.Join(strs, " ")))
  }
}

func WriteArray(filename string, arr []float64) {
  f, err := os.Create(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()
  for i := 0; i < len(arr); i++ {
    f.WriteString(fmt.Sprintf("%f\n", arr[i]))
  }
}

func WriteInts(filename string, arr []int) {
  f, err := os.Create(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()
  for i := 0; i < len(arr); i++ {
    f.WriteString(fmt.Sprintf("%d\n", arr[i]))
  }
}

func main() {
  data := GetData("data.txt")
  K := 2
  S := 16

  r := rand.New(rand.NewSource(20))

  P0 := make([][]float64, K)
  E0 := make([][]float64, K)
  nu0 := make([]float64, K)
  for k := 0; k < K; k++ {
    nu0[k] = 1.0 / float64(K)
    P0[k] = make([]float64, K)
    sum := 0.0
    for j := 0; j < K; j++ {
      tmp := r.Float64()
      P0[k][j] = tmp
      sum += tmp
    }
    for j := 0; j < K; j++ {
      P0[k][j] /= sum
    }
    E0[k] = make([]float64, S)
    sum = 0.0
    for j := 0; j < S; j++ {
      tmp := r.Float64()
      E0[k][j] = tmp
      sum += tmp
    }
    for j := 0; j < S; j++ {
      E0[k][j] /= sum
    }
  }

  P, E, nu, logprobs := hmm.BaumWelch(data, P0, E0, nu0)
  WriteMatrix("P.txt", P)
  WriteMatrix("E.txt", E)
  WriteArray("nu.txt", nu)
  WriteArray("logprobs.txt", logprobs)

  vit := hmm.Viterbi(data, P, E, nu)
  WriteInts("viterbi.txt", vit)

  hid, obs := hmm.GenerateSequence(50, P, E, nu, r)
  fmt.Printf("hid: %v\n", hid)
  fmt.Printf("obs: %v\n", obs)
}
