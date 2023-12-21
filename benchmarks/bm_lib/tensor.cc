#include "tensor.h"

int accumulate(std::<int>val) {
  int res = 1;
  for (auto i : val) {
    res *= i;
  }
  return res;
}

tensor(std::string name, std::<int>shape) {
  name_ = name;
  shape_ = shape;
  cudaMalloc(&data_, sizeof(half) * accumulate(shape))
}

~tensor() { cudaFree(data_); }
