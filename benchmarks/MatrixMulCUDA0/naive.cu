// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA0/naive.cuh"

template <typename T>
__global__ void gemv_kernel(T *A, T *B, T *C, int m, int n, int k) {
  // Compute thread ID and corresponding matrix element
  long int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > m * n) return;

  int x = tid % m;
  int y = tid / m;

  if (x < m && y < n) {
    // Compute dot product of row of A and column of B
    T value = 0;
    for (int i = 0; i < k; i++) {
      value += A[x * k + i] * B[i * n + y];
    }
    // Update matrix C
    C[x * n + y] = value;
  }
}

template <size_t threadsPerBlock, typename T>
void GEMV0(T *dA, T *dB, T *dC, int m, int n, int k) {
  int blocks = ceil((float)m * n / threadsPerBlock);

  gemv_kernel<<<blocks, threadsPerBlock>>>(dA, dB, dC, m, n, k);
  cudaDeviceSynchronize();
}

template void GEMV0<TPB, float>(float *dA, float *dB, float *dC, int m, int n,
                                int k);
