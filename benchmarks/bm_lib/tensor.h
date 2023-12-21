#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "utils.h"

class tensor {
 public:
  void* data_;
  std::vector<int> shape;
  std::string name;

  tensor(std::string name, std::<int>shape);
  ~tensor();
};
