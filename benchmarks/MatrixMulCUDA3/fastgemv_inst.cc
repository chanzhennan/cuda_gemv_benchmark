// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA3/fastgemv_inst.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_base/basegemv.h"

template <typename T>
class FastGemvInst : public BaseGemv<T> {
  using BaseGemv<T>::getDeviceA;
  using BaseGemv<T>::getDeviceB;
  using BaseGemv<T>::getDeviceC;

 public:
  void callKernel(benchmark::State &state) override {
    GEMV3<T>(getDeviceA(), getDeviceB(), getDeviceC(), state.range(0),
             state.range(1), state.range(2));
  }
};

#define BENCHMARK_GEMV3_OP(name, dType)                                      \
  BENCHMARK_TEMPLATE_DEFINE_F(FastGemvInst, name, dType)                     \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(FastGemvInst, name)                                   \
      ->Unit(benchmark::kMillisecond)                                        \
      ->ArgsProduct({{1}, {4096, 11008}, {4096, 11008}});

#define BENCHMARK_GEMV3_OP_TYPE(dType) BENCHMARK_GEMV3_OP(GEMV_##dType, dType)

BENCHMARK_GEMV3_OP_TYPE(half)
