// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "MatrixMulCUDA1/fastgemv.cuh"
#include "bm_base/basegemv.h"

template <typename T>
class QuantGemv : public BaseGemv<T> {
  using BaseGemv<T>::getDeviceA;
  using BaseGemv<T>::getDeviceB;
  using BaseGemv<T>::getDeviceC;

 public:
  void callKernel(benchmark::State &state) override {
    GEMV2<T>(getDeviceA(), getDeviceB(), getDeviceC(), state.range(0),
             state.range(1), state.range(2));
  }
};

#define BENCHMARK_GEMV2_OP(name, dType)                                      \
  BENCHMARK_TEMPLATE_DEFINE_F(QuantGemv, name, dType)                        \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(QuantGemv, name)                                      \
      ->Unit(benchmark::kMillisecond)                                        \
      ->ArgsProduct({{1}, {4096, 11008}, {4096, 11008}});

#define BENCHMARK_GEMV2_OP_TYPE(dType) BENCHMARK_GEMV2_OP(Gemv_##dType, dType)

BENCHMARK_GEMV2_OP_TYPE(half)
