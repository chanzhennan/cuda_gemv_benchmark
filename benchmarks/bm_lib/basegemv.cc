#include "basegemv.h"

template <typename T>
void BaseGemv<T>::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

template <typename T>
void BaseGemv<T>::SetUp(const ::benchmark::State &state) {
  // Populate array
  unsigned long M = (unsigned long)state.range(0);
  unsigned long N = (unsigned long)state.range(1);
  unsigned long K = (unsigned long)state.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;
  unsigned long csize = M * N;

  cudaMallocManaged(&dA, sizeof(T) * asize);
  cudaMallocManaged(&dB, sizeof(T) * bsize);
  cudaMallocManaged(&dC, sizeof(T) * csize);
  cudaMallocManaged(&testC, sizeof(T) * csize);

  genData(state);
}

template <typename T>
void BaseGemv<T>::genData(const ::benchmark::State &st) {
  unsigned long M = (unsigned long)st.range(0);
  unsigned long N = (unsigned long)st.range(1);
  unsigned long K = (unsigned long)st.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;

  cudabm::genOnes<T>(dA, asize);
  cudabm::genOnes<T>(dB, bsize);
}

template <typename T>
T *BaseGemv<T>::getDeviceA() {
  return dA;
}

template <typename T>
T *BaseGemv<T>::getDeviceB() {
  return dB;
}

template <typename T>
T *BaseGemv<T>::getDeviceC() {
  return dC;
}

template <typename T>
T *BaseGemv<T>::getDeviceTestC() {
  return testC;
}

template <typename T>
void BaseGemv<T>::verify(const ::benchmark::State &st) {
  // for test M, N, K = state.range(0)

  cudabm::Gemm<T>(dA, dB, testC, st.range(0), st.range(1), st.range(2));
  // if (!cudabm::Equal<T>(st.range(0) * st.range(1), dC, testC, 1e-2))
  //   throw std::runtime_error("Value diff occur in Dense");
}
template <typename T>
void BaseGemv<T>::TearDown(const ::benchmark::State &st) {
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(testC);
}
template <typename T>
double BaseGemv<T>::getDataSize(const ::benchmark::State &state) {
  // datasize = 2 * M * N
  return (double)(2 * state.range(0) * state.range(1));
}

template <typename T>
double BaseGemv<T>::getFlops(const ::benchmark::State &state) {
  // flops =  2 * M * N * K / s
  return (double)(2 * long(state.range(0)) * state.range(1) * state.range(2));
}

template class BaseGemv<float>;
template class BaseGemv<half>;
