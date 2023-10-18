#include "basegemv.h"

void BaseGemv::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

void BaseGemv::SetUp(const ::benchmark::State &state) {
  // Populate array
  unsigned long M = (unsigned long)state.range(0);
  unsigned long N = (unsigned long)state.range(1);
  unsigned long K = (unsigned long)state.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;
  unsigned long csize = M * N;

  cudaMallocManaged(&dA, sizeof(float) * asize);
  cudaMallocManaged(&dB, sizeof(float) * bsize);
  cudaMallocManaged(&dC, sizeof(float) * csize);
  cudaMallocManaged(&testC, sizeof(float) * csize);

  genData(state);
}

void BaseGemv::genData(const ::benchmark::State &st) {
  unsigned long M = (unsigned long)st.range(0);
  unsigned long N = (unsigned long)st.range(1);
  unsigned long K = (unsigned long)st.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;

  cudabm::genOnes(dA, asize);
  cudabm::genOnes(dB, bsize);
}

float *BaseGemv::getDeviceA() { return dA; }

float *BaseGemv::getDeviceB() { return dB; }

float *BaseGemv::getDeviceC() { return dC; }

float *BaseGemv::getDeviceTestC() { return testC; }

void BaseGemv::verify(const ::benchmark::State &st) {
  // for test M, N, K = state.range(0)
  cudabm::Gemm(dA, dB, testC, st.range(0), st.range(1), st.range(2));
  if (!cudabm::Equal<float>(st.range(0) * st.range(1), dC, testC, 1e-2))
    throw std::runtime_error("Value diff occur in Dense");
}

void BaseGemv::TearDown(const ::benchmark::State &st) {
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(testC);
}

double BaseGemv::getDataSize(const ::benchmark::State &state) {
  // datasize = 2 * M * N
  return (double)(2 * state.range(0) * state.range(1));
}

double BaseGemv::getFlops(const ::benchmark::State &state) {
  // flops =  2 * M * N * K / s
  return (double)(2 * long(state.range(0)) * state.range(1) * state.range(2));
}
