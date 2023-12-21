
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "utils.h"

class Quant {
 public:
  Quant(int in, int out, int gs, int maxlen, int len)
      : infeature_(in),
        outfeature_(out),
        groupsize_(gs),
        max_length_(maxlen),
        length_(len) {}

  void write_4bit_tensor(tensor* src, tensor* dst);

  void pseudo_quantize_tensor();

 private:
  int infeature_;
  int outfeature_;
  int groupsize_;
  int max_length_;
  int length_;
}
