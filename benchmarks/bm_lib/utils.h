// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#define checkCuda(call)                                                 \
  do {                                                                  \
    cudaError_t status = (call);                                        \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " (" \
                << status << ") at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                           \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define checkCuBlasErrors(func)                                               \
  {                                                                           \
    cublasStatus_t e = (func);                                                \
    if (e != CUBLAS_STATUS_SUCCESS)                                           \
      printf("%s %d CuBlas: %s", __FILE__, __LINE__, _cuBlasGetErrorEnum(e)); \
  }

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);

void genRandom(float* vec, unsigned long len);

template <typename T>
void genOnes(T* vec, unsigned long len);

template <typename T>
void Print(T* vec, size_t len);

template <typename T>
float Sum(T* vec, size_t len);

template <typename T>
void Gemm(T* dA, T* dB, T* dC, int m, int n, int k);

template <typename T>
void transpose(T* dsrc, T* ddst, int src_m, int src_n);

template <typename Type>
bool Equal(const unsigned int n, const Type* x, const Type* y,
           const Type tolerance);

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int m, int n, int k, T* A, T* B,
                   T* C, int lda, int ldb, int ldc, S* alpha, S* beta,
                   int algo);

}  // namespace cudabm
